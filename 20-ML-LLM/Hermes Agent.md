---
tags: [hermes-agent, agent, nous-research, memory, skills, tool-calling, LLM, ML]
status: growing
level: 4
tier: 3
---

# Hermes Agent

**Hermes Agent** is an open-source (MIT licence) [[Agent]] program made by Nous Research. You give it a goal in a terminal or a chat app, and it asks a [[LLM]] what to do, runs tools (shell commands, files, web, browser) and loops until the goal is done. What makes it different from a plain agent loop is that it keeps notes about you and about how it solved things, so the next task starts with more than the last one did. It is a separate thing from the "Hermes" family of language models by the same company: Hermes Agent is the program around a model, and it can use many models.

## 🟢 1. Simple (age 12)
Imagine a helper who can use a computer for you. You say "check my server and tell me if it is fine", and the helper types the commands, reads the answers and reports back. A normal helper forgets everything when you close the window. This helper writes a small notebook (who you are, what you like) and a recipe box (how it solved things before), and reads both each morning. So it gets better at your jobs over time.

**Example:** the first time you ask it to make a weekly report, it works out the steps slowly. It saves them as a recipe. Next week it follows the recipe and only fixes what changed.

## 🔵 2. How it works
1. **You talk to it** through a terminal screen (`hermes`), a desktop app, a messaging bot (Telegram, Discord, Slack and others), or an editor plugin. All of these feed the same agent object.
2. **It builds a prompt:** the [[System Prompt]], your saved memory notes, the list of available skills, and the conversation so far.
3. **It calls a model** through a provider you choose (a hosted service, or a model running on your own computer). The model answers with text or with a request to use a tool ([[Tool Calling]]).
4. **It runs the tool** (for example a shell command), puts the result back into the conversation and asks the model again. This repeats until the model gives a final answer ([[ReAct]]-style loop).
5. **It remembers.** Two small text files hold notes about the agent's work and about you, and every past conversation is stored in a searchable database.
6. **It writes skills.** After a task that was worth repeating, the agent can save the steps as a skill: a folder with a `SKILL.md` file that it can reload later.
7. **It can run alone on a schedule** ("every 2h, check the server"), and it can start helper agents for side jobs.
8. **Commands run in a place you pick:** your own machine, a Docker container, a remote machine over SSH, or a cloud sandbox ([[Docker]]).

No image: I did not search for a free picture of Hermes Agent, and the diagrams on its docs site were not checked for a free licence. The facts below come from the official docs text instead.

## 🟠 3. Technical detail
Everything in this section comes from the official docs (hermes-agent.nousresearch.com/docs, pages: home, architecture, memory, skills, tools, cron) and the GitHub README, read through a fetch tool. The skills and cron pages were cut off by the fetch, so their later parts reached me as a summary, not as text. **I did not install or run Hermes Agent**, so nothing below is measured by me.

**One agent, many doors.** The docs say the command-line app, the messaging gateway, the editor protocol (ACP, for VS Code, Zed and JetBrains), a batch runner, an API server and a Python library all use one `AIAgent` class (in `run_agent.py`). The loop itself is described as synchronous. A conversation turn goes: user input, build the system prompt, resolve which provider to call, call it (one of three API styles: `chat_completions`, `codex_responses`, `anthropic_messages`), run any requested tool, loop, then save the session.

**Design principles named on the architecture page:** *prompt stability* (the system prompt does not change in the middle of a conversation, which is what lets [[Prompt Caching]] work, my reading, not stated that way on the page), *observable execution* and *interruptible*.

**Memory, in three layers.**

| Layer | Where | Size or form | How it is used |
|---|---|---|---|
| Agent notes | `~/.hermes/memories/MEMORY.md` | 2,200 characters (about 800 tokens) | Put in the system prompt as a frozen snapshot at the start of a session |
| User profile | `~/.hermes/memories/USER.md` | 1,375 characters (about 500 tokens) | Same |
| Past sessions | `~/.hermes/state.db` (SQLite with full-text search) | every message | Looked up on demand with the `session_search` tool |

- **Layer columns:** *Where* is the file on disk. *Size* is the limit the docs state; going over returns an error and nothing is compacted automatically. *How it is used* says whether it is always in the prompt or fetched when needed.
- Memory edits are made with a `memory` tool (add, patch, remove). A write goes to disk at once but only shows up in the prompt in the **next** session, which keeps the prompt stable.
- The docs say entries are scanned for prompt-injection and data-leak patterns ([[Prompt Injection]]).
- **Plugins** can add outside memory systems (the docs name several, for example Honcho and Mem0).

**Skills.** A skill is a folder in `~/.hermes/skills/` with a `SKILL.md` (YAML header plus instructions) and optional `references/`, `templates/` and `scripts/` folders. The model is shown only a short list first, then loads one skill's text, then a sub-file if needed (the docs call this progressive disclosure, which saves [[Context Window]] space). The agent creates and edits skills with a `skill_manage` tool, and the docs say it does so after a multi-step workflow worth repeating, after errors, or after being corrected. A `/learn` command builds a skill from a folder, a URL or a description. An option can hold new skills for your approval first. The docs say skills follow the open `agentskills.io` standard (not checked by me).

**Tools.** Built-in tools cover web search and extraction, terminal and file work, a browser, image and speech, a to-do list, running code, `delegate_task` (subagents), memory, session search and scheduling. They are grouped into toolsets you can switch on or off. Tools from [[Model Context Protocol]] servers and plugins can be added. Long-running commands can run in the background and be polled.

**Terminal backends:** local, Docker, SSH, Modal, Daytona and Vercel Sandbox, plus plugin backends such as Singularity.

**Scheduling.** The messaging gateway checks a job list every 60 seconds. Jobs are made in plain language or with `hermes cron create "every 2h" "..."`. Each run is a **fresh session with no memory of earlier runs**, so the prompt must stand alone. Output files go under `~/.hermes/cron/output/`.

**Install, as the docs give it (not run by me):** `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash` on Linux, macOS or WSL2, and a PowerShell one-liner for Windows. Piping a script from the web into a shell runs whatever it contains, so read it first.

**Where the pages disagree (reported as found, not resolved):**
- The tool count is "60+" on the docs home and "70+ ... across about 28 toolsets" on the architecture page.
- The terminal backends are "6 built-in plus plugins" on the docs home and "7 backends" in the architecture tree.
- The README says session search uses "LLM summarization", while the memory page says it returns the stored messages with no summarization.

**Summary: what the docs show.** Hermes Agent is a standard tool-calling agent loop wrapped in four things: file-based memory, a searchable history, self-written skills and a scheduler, all kept as ordinary files and one SQLite database on your disk. Caution: this is the project's own description. I did not run it, I did not read its source code, and I found no independent measurement that its "learning loop" makes it better at tasks.

## 🔴 4. Deep / formal
No equation: it is a software system, and the behaviour is the loop above, not a formula.

**Why the memory is so small.** Memory is injected into every prompt, so each character costs tokens on every call and takes space in the [[Context Window]]. The 2,200 and 1,375 character limits force the agent to keep only what matters, and the full history stays outside the prompt, searched only when needed. This is the same trade-off as [[RAG]] (keep much outside, fetch a little), applied to the agent's own past. (My reading; the docs state the limits, not this reason.)

**Skills versus memory versus fine-tuning.** Memory holds facts ("the user prefers short answers"). A skill holds a procedure ("how to deploy this app"). Neither changes the model's weights: the "learning" is text that is loaded into the prompt later, not [[Fine-tuning]]. The docs also mention saving conversations in a training-data format (ShareGPT) for others to train on, which is a separate step. Whether the text-only learning loop gives a measurable gain is not shown on the pages I read.

**Risks and limits:**
- An agent that runs shell commands and reads web pages can be steered by hostile text in those pages ([[Prompt Injection]], [[Agent Safety]]). Running commands in a Docker or cloud sandbox limits the damage; the local backend runs on your own machine.
- Skills and memory the agent writes itself can store a mistake and repeat it. The docs offer an approval switch for writes (off by default, as I read it).
- Skills downloaded from a public hub are other people's instructions; the docs say installs are scanned, and I did not test that.
- A frozen memory snapshot means a fact learned today is not in the prompt until the next session.
- Scheduled runs have no memory of earlier runs.

**What came before, and how this improved on it:**
- **Before:** a tool-calling loop where the model asks for a tool, gets the result and answers ([[ReAct]], [[Tool Calling]]), usually with only the current conversation as memory, living inside one app (an IDE helper or a chat window).
- **Its limit:** when the session ends the agent forgets, every task is solved from scratch, and the agent works only while a person is typing.
- **What changed:** persistent memory files, a searchable history, skills the agent writes for itself, a scheduler, a messaging gateway and sandbox backends, all around the same loop.
- **How much better:** I have no number. The docs and README claim a built-in learning loop but I found no benchmark in the pages I read, and I did not run it. Treat "better" as a design claim, not a measured result.

## Related
[[Agent]] · [[Tool Calling]] · [[ReAct]] · [[Model Context Protocol]] · [[Multi-Agent Systems]] · [[Agent2Agent]] · [[Agent Safety]] · [[Prompt Injection]] · [[System Prompt]] · [[Prompt Caching]] · [[Context Window]] · [[RAG]] · [[Fine-tuning]] · [[Docker]] · [[Ollama]] · [[llama.cpp]] · [[vLLM]] · [[LLM]]

## Sources
- Official docs read through a fetch tool: hermes-agent.nousresearch.com/docs (home), `/docs/developer-guide/architecture`, `/docs/user-guide/features/memory`, `/docs/user-guide/features/skills` (cut off, later part summarised), `/docs/user-guide/features/tools`, `/docs/user-guide/features/cron` (cut off, later part summarised), and the GitHub README (github.com/NousResearch/hermes-agent; MIT licence shown on the page, LICENSE file not read).
- Not read: the agent-loop internals page, the prompt-assembly page, the delegation page, the provider list, the source code. No release version or date was found on the pages. No individual authors were named ("Built by Nous Research").
- Several blog posts turned up in a web search and disagreed on release dates and versions; I did not use them for facts.
- Not done: I did not install or run Hermes Agent, and the note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Hermes Agent:::Open-source (MIT) agent program by Nous Research: a tool-calling loop plus persistent memory, searchable session history, self-written skills, a scheduler and a messaging gateway (from its docs; I did not run it).

#flashcards/ml/concept #flashcards/topic/agents-rag Is Hermes Agent the same thing as the Hermes language models?::No. Hermes Agent is the program around a model (loop, tools, memory, skills). It can use many providers and models; the Hermes models are a separate family from the same company.

#flashcards/ml/concept #flashcards/topic/agents-rag What are the steps of one Hermes Agent turn?::Build the system prompt (memory, skills, history), call the chosen model provider, run any tool the model requests, feed the result back and repeat until a final answer, then save the session (architecture page of its docs).

#flashcards/ml/concept #flashcards/topic/agents-rag What are the three layers of Hermes Agent memory?::Agent notes (MEMORY.md, 2,200 characters), a user profile (USER.md, 1,375 characters), both frozen into the prompt at session start; and every past session in a SQLite database searched on demand with session_search (docs, not run by me).

#flashcards/ml/concept #flashcards/topic/agents-rag Why is a memory edit not visible in the prompt until the next session?::The memory is injected as a frozen snapshot at session start to keep the system prompt stable; writes hit disk at once but only appear in the next session's prompt (docs). That the stability helps prompt caching is my reading.

#flashcards/ml/concept #flashcards/topic/agents-rag What is a Hermes skill and how is it used?::A folder in ~/.hermes/skills/ with a SKILL.md (YAML header plus instructions) and optional reference, template and script folders. The model sees a short list first and loads one skill's text only when needed (progressive disclosure), and the agent writes skills itself with a skill_manage tool.

#flashcards/ml/concept #flashcards/topic/agents-rag Does Hermes Agent's "learning" change the model's weights?::No, as the docs describe it: the learning is text (memory notes and skills) loaded into later prompts, not fine-tuning. Whether this improves task results was not shown on the pages I read.

#flashcards/ml/concept #flashcards/topic/agents-rag How do scheduled Hermes Agent jobs behave?::The gateway checks the job list every 60 seconds; each run is a fresh session with no memory of earlier runs, so its prompt must be self-contained (cron docs page).

#flashcards/ml/concept #flashcards/topic/agents-rag Where can Hermes Agent run its commands?::Local machine (default), Docker, SSH, Modal, Daytona or Vercel Sandbox, plus plugin backends such as Singularity. Local runs on your own machine, so a sandbox limits the damage of hostile instructions.

#flashcards/ml/concept #flashcards/topic/agents-rag What risks come with a self-writing, shell-running agent like Hermes Agent?::Prompt injection from web pages or files it reads, mistakes saved into memory or skills and repeated, untrusted skills from a public hub, and local-backend commands acting on your real machine. The docs offer write-approval switches and install scanning (not tested by me).

#flashcards/ml/concept #flashcards/topic/agents-rag Where do the Hermes Agent docs disagree with themselves?::Tool count ("60+" on the home page, "70+ across about 28 toolsets" on the architecture page), terminal backends (6 built-in plus plugins against 7), and whether session search uses LLM summarization (README says yes, memory page says no).

#flashcards/ml/cloze #flashcards/topic/agents-rag Hermes Agent keeps ==facts== in memory files, ==procedures== in skills, and all past conversations in a ==SQLite== database.
