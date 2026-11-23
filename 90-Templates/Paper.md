---
tags: [paper]
link: 
authors: 
year: 
status: unread
field: 
---

# {{title}}

> **In one line for a 12-year-old:** 

*Reference note for this layout: [[Attention Is All You Need 2017]]. Delete the italic hints when you fill a section in. Mark every claim "(paper, section)", "(general knowledge)" or "(mine)". Never state a reason the paper does not give as if it did.*

## The problem
What was hard, in one or two sentences. Say what the paper claims to fix.

## What people did before
The older method(s), as [[links]], and the limit of each.

## The idea
The new trick in plain words, as a short story or analogy. Name key terms as [[links]].

## The architecture (figures)
*Skip this section for a paper with no model (for example a theory or data paper) and write "The method, step by step" instead.*

Start with one paragraph on the whole design (how many parts, how they connect, the sizes). Then one sub-section per big block (for example encoder, decoder). Inside each, **one bullet per part**, each with two sub-bullets:
- *What it does:* in plain words, with the paper's numbers.
- *Why it matters:* the reason. If the paper gives none, say so and give the evidence it does have (an ablation row) or mark the reason "(general knowledge)".

Answer the "why this number and not another" questions here (why 6 layers, why that width) using the paper's ablation table, and say when the paper tried only some values.

Put each figure right under the paragraph that explains it, then the step-by-step text. Use free figures only (see CLAUDE.md). Caption: what it shows, author, licence, "a redrawing of the paper's Figure N, not the paper's own", and what differs from the paper's figure. Add a size check (parameter counts) in a short last sub-section. Architecture tables go here next to the part they describe.

## The core mechanism, step by step
*One sub-section per mechanism a reader must really understand (for Attention: scaled dot-product attention, then multi-head attention).* Each has: the idea in a paragraph or analogy, the figure, then numbered steps, each with *What it does* and *Why it matters*, then a **worked example with tiny numbers** (computed in Python), then a shapes table if matrices are involved.

## The equations
One display equation per key equation of the paper, in the order the paper uses them. Each is followed by **Where:** (a bullet per symbol), **Matters:** (what breaks without it) and **Example (tiny numbers):**. Give the paper's equation number or section.

## What they found
The headline numbers, each marked "abstract", "paper" or "memory". Then **one sub-heading per results table**, for example "Translation quality and cost (paper's Table 2)". Under each:
1. the caption line, then the Markdown table, written by me, an AI, from the paper's table (say which rows or columns were left out);
2. **What each column is, and why it matters:** a bullet per column (units, how it is measured, why the paper cares), and for row labels or symbols that are not obvious;
3. **Summary: what the table shows:** which row is best and which is worst, how the main rows compare, where the winner is not the best, and the caution (single runs, noise, omitted rows).

Say plainly when the paper contradicts itself, or when a cell could not be read.

## Questions I had, answered
*Optional. Plain-language questions that come up while reading (why is this masked, what does shifted mean, how is this different from the normal version), each with a short clear answer, a tiny example and the source mark.*

## Why it mattered
What changed afterwards.

## Limits and open questions
What it does badly, what it costs, what the authors say they did not do.

## Concepts used
- [[]]

## Reading trail
- **Came from:** [[]]
- **Led to:** [[]]

## How much I read
Which pages were read as text, which as pictures, which not (references, appendix). Which numbers are "paper", "abstract", "memory", or my arithmetic. What was not checked. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
*Write these last, at the very end of the note (see CLAUDE.md for the syntax and tags). A paper note has cards for:*
- *the problem, the idea, what came before and the key results;*
- *each part of the architecture (what it does and why);*
- *each step of each core mechanism, and its worked example;*
- *each equation: a formula card, an example card and "why this term" cards (why this power, constant or operation, what breaks without it, with a number);*
- *each embedded image (an image card);*
- *each results table (best, worst, the caution);*
- *each answered question from "Questions I had".*
