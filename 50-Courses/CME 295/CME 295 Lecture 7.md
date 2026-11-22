---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 7
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 7 slides
status: growing
---

# CME 295 Lecture 7: Retrieval, tool use and agents

> **In one line:** Shows how to give an LLM knowledge it was not trained on (retrieval-augmented generation), the ability to act (tool calling and a standard protocol), and the loop that ties it together (agents). (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **Recap.** Reasoning models, the DeepSeek recipe, and a variant of GRPO called DAPO.
2. **RAG, the motivation.** The knowledge of an LLM is limited to its training data, and its context is limited. See [[RAG]].
3. **RAG, the three stages.** Retrieve relevant documents by similarity, augment the prompt, generate. First a knowledge base is built (collect, chunk, embed), with hyperparameters such as the embedding size and chunk size. See [[Embedding Model]], [[Vector Database]] and [[Cosine Similarity]].
4. **Retrieval in two steps.** Candidate retrieval with embeddings (with extensions such as mitigating the difference between the nature of questions and documents, and contextualising chunks), then ranking with a cross-encoder re-ranker. See [[BM25]], [[Contextual Retrieval]] and [[Prompt Caching]].
5. **Measuring retrieval.** Whether the right chunk was found, and reciprocal rank at k. See [[Retrieval Metrics]].
6. **Tool calling.** Let the LLM pick a function and its arguments, run the function, and pass the result back. Methods to teach a model to use tools, common use cases, and handling many tools and tool selection. See [[Tool Calling]].
7. **A standard protocol.** Why a standard avoids duplicated work, and MCP (the host, client and server picture, with Claude Desktop as an example). See [[Model Context Protocol]] and [[Agent2Agent]].
8. **Agents.** An agent as a system that works towards a goal on its own, and ReAct (Reason plus Act). See [[Agent]] and [[ReAct]].
9. **Safety.** Risks of tools in the real world, such as data exfiltration, and remedies. See [[Agent Safety]].

## Key ideas to remember
- RAG = retrieve relevant text, add it to the prompt, then generate ([[RAG]]).
- Retrieval is usually two steps: fast candidate search with embeddings, then careful re-ranking ([[Embedding Model]], [[BM25]]).
- Tool calling lets a model take actions; a standard protocol (MCP) avoids rewriting every tool for every model ([[Tool Calling]], [[Model Context Protocol]]).
- An agent is an LLM in a loop of thinking, acting and observing ([[Agent]], [[ReAct]]).

## Papers named in the lecture
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", Lewis et al., 2020 (named on the slides; [[RAG]])
- Cross-encoders, SBERT.net (suggested reading on a slide)
- "ToolSword" (Ye et al., 2024) and "Towards Tool Use Alignment of Large Language Models" (Chen et al., 2024), cited on the safety slides ([[Agent Safety]])
- "ReAct: Synergizing Reasoning and Acting in Language Models", Yao et al. ([[ReAct]])

## Questions I still have
- How big should a chunk be, and when does a re-ranker pay for itself?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 6]]. Lecture after: [[CME 295 Lecture 8]].
- Concept notes this lecture feeds: [[RAG]], [[Embedding Model]], [[Vector Database]], [[BM25]], [[Contextual Retrieval]], [[Prompt Caching]], [[Retrieval Metrics]], [[Tool Calling]], [[Model Context Protocol]], [[Agent2Agent]], [[Agent]], [[ReAct]], [[Agent Safety]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: what does it cover?::A recap (DAPO, a GRPO variant), RAG, retrieval in two steps, measuring retrieval, tool calling, a standard protocol (MCP), agents, and safety.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: what are the three stages of RAG?::Retrieve relevant documents by similarity, augment the prompt, generate. First a knowledge base is built (collect, chunk, embed), with hyperparameters such as embedding size and chunk size.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: what are the two steps of retrieval?::Candidate retrieval with embeddings (with extensions such as contextualising chunks), then ranking with a cross-encoder re-ranker.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: how is retrieval measured?::Whether the right chunk was found, and reciprocal rank at k.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: what is tool calling?::The LLM picks a function and its arguments, the function runs, and the result is passed back.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: why a standard protocol such as MCP?::A standard avoids duplicated work (every tool rewritten for every model). The slides show the host, client and server picture with Claude Desktop as an example.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: what is an agent and what is ReAct?::An agent is a system that works towards a goal on its own, an LLM in a loop of thinking, acting and observing. ReAct is Reason plus Act.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: what safety risks does it name?::Risks of tools in the real world, such as data exfiltration, and remedies.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: which papers are named?::RAG (Lewis et al., 2020, named on slides), cross-encoders (SBERT.net, suggested reading), ToolSword (Ye et al., 2024) and "Towards Tool Use Alignment of Large Language Models" (Chen et al., 2024) on the safety slides, and ReAct (Yao et al.). Most were named only and not read.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 7: the open question the note raises?::How big should a chunk be, and when does a re-ranker pay for itself?

#flashcards/courses/concept #flashcards/topic/course-cme295 How was this CME 295 lecture note built, and what was not seen?::From the lecture slides (PDF) read as extracted text. The note did not see the figures, any speaker notes or the recordings, and did not open the course playlist. Papers only "named on the slides" were not read. The lecture title is the note's own label, not the instructors'.
