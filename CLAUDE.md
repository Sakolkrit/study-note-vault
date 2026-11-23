# CLAUDE.md

This folder is an **Obsidian vault**: a personal knowledge base for AI, math, LLMs and programming. The owner is learning these fields and wants notes that start at a 12-year-old's level and get steadily deeper. Open `README.md` for the short user-facing guide; this file is for Claude.

## Vault layout

| Folder | Holds |
|---|---|
| `00-Inbox` | new, unsorted notes (Obsidian creates new notes here) |
| `10-Math` | linear algebra, calculus, probability, optimization |
| `20-ML-LLM` | transformers, RAG, training, agents, papers |
| `30-Programming` | python, libraries (NumPy, pandas, PyTorch, TensorFlow), tooling, snippets |
| `40-Algorithms` | data structures, algorithms, Big-O and complexity, with runnable code |
| `50-Courses` | one folder per course: an index note plus one note per lecture (summaries that link to concept notes); from `90-Templates/Course.md` and `Lecture.md` |
| `55-Books` | one folder per book: an index note plus one note per chapter (summaries that link to concept notes); from `90-Templates/Book.md` and `Chapter.md` |
| `60-Projects` | one note per project, from `90-Templates/Project.md`: goal, result, exact steps that were run, what went wrong, what was not run |
| `70-Research` | one note per research paper, from `90-Templates/Paper.md`; named "Short Name Year" (e.g. `AlexNet 2012`) |
| `90-Templates` | `Concept.md`, `Lecture.md`, `Course.md`, `Book.md`, `Chapter.md`, `Project.md`, `Paper.md` |
| `99-Attachments` | images and PDFs |

Put a concept in the folder of its home field (`Vector` in `10-Math`, `Attention` in `20-ML-LLM`, `Pandas` in `30-Programming`, `Big-O` in `40-Algorithms`). `50-Courses` and `60-Projects` were started with pilots (CME 295 and Machine Learning Specialization as courses; Iris Classifier Service, Heart Disease Classifier and Vault Search with RAG as projects). Do not edit `.obsidian/` unless asked.

## Concept notes (the main thing you write)

Start from `90-Templates/Concept.md`. One concept = one note, named by the term.

**Frontmatter:** `tags` (topic families, e.g. `[RAG, LLM, transformer]`; include every parent field), `status` (`seed` → `growing` → `done`), `level` (deepest level written, 1 to 4), `tier` (how big or advanced the topic itself is: 1 foundation such as Vector or Python, 2 core method or tool, 3 advanced method or system, 4 big or frontier topic such as LLM or RLHF; judged by how much you must know first). The graph view colours nodes by `tier` (green, blue, orange, red, the same colours as the level dots), so set it on every new note.

**Four levels, in this order, each harder than the last.** Each level heading carries a colour marker, so the level shows at a glance: `## 🟢 1. Simple (age 12)`, `## 🔵 2. How it works`, `## 🟠 3. Technical detail`, `## 🔴 4. Deep / formal`. Copy these headings exactly (the template has them).
1. **Simple (age 12):** one or two plain sentences, no jargon, plus an everyday **Example**.
2. **How it works:** numbered steps in plain words. Real terms appear here, each as a `[[link]]`. Put the image here.
3. **Technical detail:** architecture, algorithms, a short runnable code snippet.
4. **Deep / formal:** math, edge cases, trade-offs, limits, open problems.

Then `## Related` (links) and `## Sources`.

**History and improvement block (in level 4, before `## Related`).** Every concept note says what came before it and how it improved on that: `**What came before, and how this improved on it:**` with four bullets, *Before*, *Its limit*, *What changed*, *How much better*. Put a number measured in the note's own code in *How much better* whenever the comparison can be run (parameters, steps, accuracy, time-free counts); otherwise give the published figure and its source. Dates, names and published figures are checked against a source (arXiv or the paper's page) or flagged as from memory. Notes written before this rule get the block added as they are revisited.

**Every display equation (`$$...$$`) must be followed by two things:**
- a `**Where:**` list: one bullet per symbol saying what it is in plain words, and `**Matters:**` what it does to the result (what breaks or changes without it);
- an `**Example (tiny numbers):**` block: 2 to 3 easy numbers worked step by step to the final answer, ending with what the answer means.

If a formula appears inline in a sentence and the symbols are not obvious, promote it to a display equation and give it the same two blocks.

## Course notes (`50-Courses`)
- A course gets a folder, an index note named after the course (`CME 295`) and one note per lecture (`CME 295 Lecture 1`).
- The index lists logistics (only what the source shows), a lecture map, how the lectures fit together, and what was not checked. A lecture note says what the lecture covers in order, the few key ideas, the papers named, open questions, and exactly what it was built from.
- **Lecture notes summarise and link; the detail lives in concept notes.** Do not copy a concept's four levels into a lecture note. Link to it, and write the concept note if it does not exist.
- Never invent a lecture date, a name or a number. Say what was read (slides as text, recording, reading) and what was not. Papers only "named on the slides" are marked as not read.
- **Course flashcards are about knowledge, not about the course.** A card asks about the idea ("What does FlashAttention fix?"), never about which lecture or week taught it ("What does Lecture 4 cover?"), which papers were named, what the note was built from, the logistics, or the open questions of the note. Do not put "Lecture N" or "Week N" in a question. Keep honest marks inside the answer where they matter ("my own computation", "my rerun", "from the course slides, not read further"). Course index notes (`CME 295`, `Machine Learning Specialization`) have no flashcards section. The provenance (what was read or rerun) stays in the note body.

## Book notes (`55-Books`)
- A book gets a folder, an index note named after the book (from `90-Templates/Book.md`) and one note per chapter (from `Chapter.md`, named "Book Name Chapter N" or "Book Name Chapter N Title"). Add the book to `Books Index`.
- **Same rule as lectures: chapter notes summarise and link; the detail lives in concept notes.** Do not copy a concept's four levels into a chapter note.
- **Only summarise what was read.** The index and every chapter note say what the text was (PDF, pasted text, photos), which pages or chapters were read in full, which were skimmed and which not read at all. A chapter that was not read gets no summary. Never invent a page, a name or a number, and never write a chapter from memory of the book.
- **Own words, short quotes.** Do not copy long passages. A short quote goes in quotation marks with its page number. Numbers from the book are marked "from the book (page N)" and numbers from my own runs "my own run".
- Papers or sources the book names but that were not read are marked "named in the book, not read".

## Project notes (`60-Projects`)
- One note per project, from `90-Templates/Project.md`: goal, a short result with measured numbers, the exact versions used, numbered steps, what went wrong, what was not run, concepts used, next ideas.
- **Build it first, then write it.** Every file and command in the steps is run in a fresh environment, and the note's code blocks are extracted and run again from the note itself before the note is called done.
- Say plainly what was not run (for example a Docker build when the engine was down) and why. Never present a step as working unless it was run.
- Keep the code inside the note. The vault holds notes, not project folders. Link every tool or idea to its concept note.

## History notes and paper notes

- `History of AI`, `History of Image Classification`, `History of Text Analytics` and `From FNN to RNN` (in `20-ML-LLM`) are timelines. Each step says what limit of the previous step it fixed, with a number from a source. Mark every date "checked" or "memory".
- **The reference paper note is `Attention Is All You Need 2017`.** Follow its layout, which `90-Templates/Paper.md` spells out: problem, what came before, idea; the architecture explained **part by part** (each part: what it does, why it matters; the "why this number and not another" questions answered with the paper's ablation evidence); the core mechanisms **step by step** with a figure placed next to its explanation, a tiny-numbers worked example and a shapes table; the equations; the results, with each table under its own sub-heading, a column explanation and a **summary of what the table shows** (best, worst, caution); "Questions I had, answered" for plain-language confusions; limits; how much was read; flashcards last.
- Paper notes in `70-Research` follow `90-Templates/Paper.md`. Say how much was read (abstract, body) and mark each number "abstract", "paper" or "memory". Link the concept notes the paper uses and the paper before and after it. Never invent authors, titles or figures.
- **Read the body, not only the abstract.** Download the arXiv PDF and extract the text (PyMuPDF in a scratch venv; set `PYTHONIOENCODING=utf-8` on Windows). Say which pages were read and which not (references, appendix).
- **Equations.** Every equation the paper uses to solve its problem goes in a section `## The equations`, as a display equation with the usual `**Where:**` (one bullet per symbol), `**Matters:**` and `**Example (tiny numbers):**` blocks (numbers computed in Python first). Give the paper's equation number or section.
- **Architecture figures.** Show the paper's own architecture, for example the Transformer model, scaled dot-product attention and multi-head attention for `Attention Is All You Need`. A paper's figures are copyrighted (arXiv's usual licence is not open, and a "may reproduce for scholarly works" line is not a free licence). So embed a figure only if the paper is CC BY or similar (check the licence link on the arXiv abstract page); then crop it from the PDF (PyMuPDF `get_pixmap(dpi=220, clip=...)`), look at the crop, name the authors and the licence in the caption, and say it is cropped from the paper. Otherwise use a free redrawing from Wikimedia Commons (rules in "Images": check licence and author, look at it, compare it with the paper's figure by eye, and say in the caption that it is a redrawing and what differs), or say that none was found. Never copy the paper's own picture without an open licence.
- **Tables go where their content belongs, not in one lump.** Put each important table (results, ablations, complexity, architecture sizes) inside the part of the note it supports: a table about the architecture goes in the architecture explanation next to the part it describes; a table comparing results goes in the results section (`What they found`) under its own sub-heading, such as "Translation quality and cost (paper's Table 2)". Do not collect them under a heading like "Tables from the paper". Each table is a Markdown table **written by me, an AI, from the paper's table text**; say so under the table, and say which rows or columns were left out. Check every number against the extracted text. Columns I add (differences, ratios) are marked "mine" and computed in Python. Say plainly when a cell could not be read from the text extraction. **After every table, explain each column**: a bullet per column saying what it is (and how it is measured, including units such as BLEU, FLOPs, top-5 error) and why it matters for the paper's claim; explain row labels and symbols that are not obvious. Mark general-knowledge definitions "(general knowledge)".
- **Cards.** A paper note's cards cover the idea, each part of the architecture, each step of the core mechanisms, each equation (a formula card, an example card and "why this term" cards: why this power, constant or operation, what breaks without it, with a number), each embedded image (an image card), each results table (best, worst, caution) and each answered question. Notice and say when the paper contradicts itself (for example two different numbers for the same result).

## Writing style

- Write normal, friendly, plain prose. Do not write notes in terse "caveman" style; that is only how Claude talks in chat.
- Levels 1 and 2 must be understandable by a 12-year-old: short sentences, everyday analogies, no unexplained symbols.
- Every hard word becomes a `[[link]]`, even if the target note does not exist yet. An unresolved link is a to-do (find them with a grep for `\[\[` and compare against existing file names).
- Be honest about uncertainty. Claims, constants, dates or authors written from memory must be flagged or checked, not stated as certain. Never invent paper titles, numbers or quotes.
- Math in `$...$` and `$$...$$`. Code in fenced blocks with a language tag. Keep snippets small and runnable.

## Images

Use real images found online, never AI-generated ones.

1. Search Wikimedia Commons through its API (namespace 6 is files), e.g. `list=search&srnamespace=6&srsearch=<topic>`.
2. Get the exact URL, license and author with `prop=imageinfo&iiprop=url|extmetadata`. Only use images that are free to reuse (CC BY, CC BY-SA, CC0, public domain). Do not guess upload URLs; take them from the API result.
3. Download with `curl -sL -A "Mozilla/5.0 study-vault"` into `99-Attachments/` using a lowercase-hyphen name (`attention-overview.svg`). Confirm the file type with `file`.
4. **Look at the image before captioning it.** The `Read` tool shows PNG and JPG. For an SVG, request a PNG thumbnail (`iiurlwidth=700`) just to view it.
5. Embed with `![[name.ext]]` and add an italic caption under it: what it shows, author, link to the Commons file page, license.
6. If no on-topic image exists, skip the image and tell the owner. Do not use an unrelated picture. On Windows set `PYTHONIOENCODING=utf-8` when piping API output through Python.

## Flashcard rules (Spaced Repetition plugin)

Every note gets a `## Flashcards` section **at the very end** (after `## Sources`), written for the community plugin *Spaced Repetition* (`st3v3nmw/obsidian-spaced-repetition`). Only one such section per note. Notes in `90-Templates/`, the index note `Interview Question Index`, the course index notes (`CME 295`, `Machine Learning Specialization`) and the seven folder index pages (`ML and LLM Index`, `Programming Index`, `Math Index`, `Algorithms Index`, `Projects Index`, `Research Index`, `Books Index`) have none. A new concept note must be added to its folder's index page, under its topic.

**Syntax**
- Single-line card: `Question::Answer`. Reversed card: `Term:::Meaning`. Cloze: `==hidden words==`.
- Multi-line card: the question lines, then a line with only `?`, then the answer lines. No blank line inside a card. Cards are separated by one blank line.
- Image card: the `![[file]]` line, then the question line, then `?`, then the answer.
- Never put `==`, `::` or a lone `?` line in note text outside the flashcards and code blocks. Never put a cloze `==` around maths, never mix `==` and `::` on one line, and never write a run of `=` signs in a card (write "a line of equals signs").
- Math inside a card is allowed (`$...$`, `$$...$$`). Check that `$` signs pair up on every line. Code spans (backticks) are safe.
- Never write a bare nested list such as `[[0, 1], [1, 0]]` in prose or in a card: Obsidian reads `[[...]]` as a link and makes a ghost node in the graph (and a stub file if clicked). Put it in backticks or write it as a LaTeX matrix. Check with a scan for `[[` outside code whose target is not a note.

**Tags: two per card, on the card's first line**
- Format: `#flashcards/<field>/<type> #flashcards/topic/<topic> Question...` (the first line of the card; for an image card, in front of the `![[...]]`). The section also starts with one line `#flashcards/<field>`.
- `<field>`: `math`, `ml`, `programming`, `algorithms`, `projects`, `research`, `courses`, `books`.
- `<type>`: `formula`, `example`, `concept`, `image`, `cloze`, `reverse`. Rule used: an image card starts with `![[`; a card with `==` is `cloze`; a `?` card whose answer starts with `$$` is `formula`; a `:::` card is `reverse`; a one-line card whose question has a digit or `$` is `example`; the rest are `concept`. Research, course and book cards are all `concept`.
- `<topic>`: one of the topics below. Use an existing topic if one fits, and add a new one only when none does. A book gets its own topic `book-<short-name>`, added to the list when the book is added. Keep the mapping (note name to topic) in one script or table so every card of a note gets the same topic.
  - math: `linear-algebra`, `calculus-optimization`, `probability`, `information-theory`
  - ML: `ml-foundations`, `classic-models`, `clustering`, `evaluation`, `data-features`, `neural-nets`, `training`, `vision`, `generative`, `transformers-nlp`, `llm`, `alignment-rl`, `llm-serving`, `agents-rag`, `history`, `time-series`, `recommenders`, `anomaly-detection`, `mlops`
  - programming: `dev-tools`, `python-core`, `data-libs`, `dl-frameworks`, `web-apps`
  - algorithms: `data-structures`, `algorithm-design`, `searching-sorting`, `graph-algorithms`, `complexity`
  - projects: `project-classic-ml`, `project-deep-learning`, `project-rl`, `project-rag`, `project-llm`, `project-agents`, `project-mlops`
  - research: `research-vision`, `research-detection-segmentation`, `research-nlp`, `research-llm`, `research-generative`, `research-time-series`
  - courses: `course-cme295`, `course-mls-1`, `course-mls-2`, `course-mls-3`
  - books: one `book-<short-name>` per book (none yet)

**What to write**
- **One card per display equation.** The answer is the `$$` formula followed by a bullet list of every symbol, what it is and what it does (the `Where:` and `Matters:` content). A second card gives the tiny-numbers example from the note.
- **One image card per embedded image**, with the answer taken from the note's own checked caption.
- Concept cards (`Term:::Meaning` and question/answer) from levels 1 to 4, cloze cards for the key phrases, and cards for the numbers in the history block. Cards for notes without equations (history, course, project, paper notes) come from the key ideas, the numbers, the limits and the "how much was read / what was not run" sections.
- Paper cards carry the note's own marks ("abstract", "paper", "memory"). Course cards ask only about the knowledge (see Course notes) and keep marks like "my rerun" inside the answer. Never state a claim more firmly than the note does.
- **Every number on a card is computed in Python first** (or copied from the note's own checked output) and the card says when a figure is "my arithmetic" or "from memory".
- Do not edit `.obsidian/`. Keep the Flashcards section in sync when the note changes.
- **Never regenerate or overwrite a note whose cards have been reviewed.** The plugin stores the review schedule inside the note, as `<!--SR:...-->` lines after the cards (one note has them so far: `Batch Normalization`). Edit such a note in place and keep those lines; rebuilding it from a template would erase the user's progress. This matters more once cards are reviewed on a phone.

**Checks after writing cards**
- Equation count equals formula-card count, and image count equals image-card count (embeds count only the note body, not the Flashcards section).
- No `\r` characters (write with LF line endings), exactly one `## Flashcards` heading, every card block has both tags, no stray `$`, `==`, `::` or lone `?` outside cards and code.
- Report plainly that the cards were not reviewed in Obsidian. The plugin counts more cards than blocks (a reverse card is two, a cloze is one per blank).

## Verify before saying done

- Run every code snippet in the note and check its printed results match what the note claims.
- Compute every number used in an **Example** with Python first, then copy the results in.
- After writing, count that each note has as many `Where:` lists as display equations, and as many `Example (` blocks as `Where:` lists.
- Say plainly what was not checked (for example, that the note was not opened in Obsidian).

## Conventions

- File names use Title Case with spaces and match `[[links]]` exactly (`Cosine Similarity.md`, `Vector Database.md`).
- Link style is wiki links (`[[Note]]`), not Markdown links, for notes inside the vault.
- A file name cannot contain `/`. Name the file without it (`AB Testing.md`) and link with an alias: `[[AB Testing|A/B Testing]]`.
- Close each notes session by listing which `[[links]]` are still unwritten and recommending the next topic.
