---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 8
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 8 slides
status: growing
---

# CME 295 Lecture 8: Evaluating LLMs and agents

> **In one line:** Asks how you tell whether a free-text answer is good: human ratings, rule-based scores, a model as the judge with its biases, and then how to evaluate agents and where they fail. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **Why evaluating is hard.** Free-form text has no single right answer, so the ideal is human ratings, which are slow and subjective.
2. **Rule-based metrics.** Write reference answers once and compare: BLEU and ROUGE, METEOR (translation scores with explicit ordering), with the limit that they ignore stylistic variation. See [[Evaluation Metrics]] and [[Perplexity]].
3. **LLM-as-a-judge.** A model reads the prompt, the response and criteria, then gives a rationale and a score. Enforcing an output format in practice, the main variations (pointwise and pairwise) and the biases: position, verbosity and self-enhancement. See [[LLM as a Judge]].
4. **Best practices and a revised workflow.** Crisp guidelines for the judge, and which dimensions to evaluate (task performance and alignment, such as usefulness, factuality, relevance, tone, style and safety). A focus on factuality.
5. **Benchmarks.** Families of common benchmarks: knowledge, reasoning, coding and safety. See [[Benchmarks]] and [[Chatbot Arena]].
6. **Evaluating agents.** The agentic mindset (observe, plan, act, using ReAct) and the failure modes of tool use: the model does not use a tool, invents a tool, uses the wrong tool or infers the wrong argument; the tool call itself errors or returns nothing or something wrong; and the final response is wrong. See [[Agent]], [[Tool Calling]] and [[Agent Safety]].

## Key ideas to remember
- Reference-based scores are cheap but miss style; humans are the gold standard but slow ([[Evaluation Metrics]]).
- A model can act as the judge, but it has biases (position, verbosity, self-enhancement) that you must test for ([[LLM as a Judge]]).
- Agents fail in specific, nameable ways, so evaluate each step of the loop, not only the final answer ([[Agent]]).

## Papers named in the lecture
- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", Zheng et al., 2023 ([[LLM as a Judge]], [[Chatbot Arena]])
- "ReAct: Synergizing Reasoning and Acting in Language Models", Yao et al. ([[ReAct]])
- "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", Jimenez et al., 2023 ([[Benchmarks]])
- "Measuring Massive Multitask Language Understanding", Hendrycks et al., 2020 ([[Benchmarks]])

## Questions I still have
- How do I check a judge model against my own judgement on a small sample?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 7]]. Lecture after: [[CME 295 Lecture 9]].
- Concept notes this lecture feeds: [[LLM as a Judge]], [[Benchmarks]], [[Chatbot Arena]], [[Evaluation Metrics]], [[Agent Safety]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 Why is evaluating free text hard?::There is no single right answer, so the ideal is human ratings, which are slow and subjective.

#flashcards/courses/concept #flashcards/topic/course-cme295 What are rule-based metrics for text and their limit?::BLEU, ROUGE and METEOR compare against reference answers written once. They ignore stylistic variation.

#flashcards/courses/concept #flashcards/topic/course-cme295 How does LLM-as-a-judge work and what are its biases?::A model reads the prompt, the response and criteria, then gives a rationale and a score. Main variations are pointwise and pairwise. The biases are position, verbosity and self-enhancement, which you must test for.

#flashcards/courses/concept #flashcards/topic/course-cme295 Which dimensions should an LLM judge evaluate?::Task performance and alignment, such as usefulness, factuality, relevance, tone, style and safety, with a focus on factuality.

#flashcards/courses/concept #flashcards/topic/course-cme295 Which benchmark families exist for LLMs?::Knowledge, reasoning, coding and safety.

#flashcards/courses/concept #flashcards/topic/course-cme295 What failure modes can an agent's tool use have?::The model does not use a tool, invents a tool, uses the wrong tool or infers the wrong argument. The tool call errors or returns nothing or something wrong. The final response is wrong. So evaluate each step of the loop, not only the final answer.
