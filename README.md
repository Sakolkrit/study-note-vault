# Vault Guide

## Folders
| Folder | Holds |
|---|---|
| 00-Inbox | new notes land here, sort later |
| 10-Math | linear algebra, calculus, probability, optimization |
| 20-ML-LLM | transformers, RAG, training, agents, papers |
| 30-Programming | python, libraries (NumPy, pandas, PyTorch, TensorFlow), tooling, snippets |
| 40-Algorithms | data structures, algorithms, Big-O, with runnable code |
| 50-Courses | courses: an index note and one summary note per lecture, linking to concept notes (first course: CME 295) |
| 55-Books | books: an index note and one summary note per chapter, linking to concept notes (summaries only of what was read) |
| 60-Projects | one note per project: goal, result, exact steps that were run, what went wrong (first: Iris Classifier Service) |
| 70-Research | one note per research paper (AlexNet, YOLO, Attention Is All You Need...) |
| 90-Templates | Concept, Lecture, Course, Book, Chapter, Project, Paper |
| 99-Attachments | images, PDFs |

## Rules
1. **One concept = one note.** Name = the term (`RAG`, `Embedding`).
2. **Progressive depth.** Concept notes go 🟢 Simple (age 12) → 🔵 How it works → 🟠 Technical → 🔴 Deep/formal (the colour dot in each heading shows the level). Stop at the level you understand; set `level:` in frontmatter and deepen later.
3. **Every hard word is a `[[link]]`.** Unwritten link = to-do (shows in graph).
4. **Tags = topic families** (`RAG`, `LLM`, `transformer`, `math`). Add every parent field the topic belongs to.
5. **Images:** real ones found online, saved to `99-Attachments`, with source URL under the image. No AI-generated images.
6. **Math** in `$...$` / `$$...$$`. **Code** in fenced blocks.
7. **Every display equation gets a `Where:` list** right under it: each symbol, what it is in plain words, and **Matters:** what it does to the result.
8. **Every `Where:` list is followed by an `Example (tiny numbers):`**: small, easy numbers worked step by step, ending with what the answer means.
9. **History block:** near the end of each concept note, **What came before, and how this improved on it** says what the older method was, its limit, what changed, and how much better (a number measured in the note's code when it can be run).

## Use
New note → `Ctrl+P` → *Templates: Insert template* → Concept / Lecture / Course / Book / Chapter / Project / Paper.

Example: [[RAG]]

## Indexes
Start here to find any note: [[ML and LLM Index]] · [[Programming Index]] · [[Math Index]] · [[Algorithms Index]] · [[Projects Index]] · [[Research Index]] · [[Books Index]] · courses: [[CME 295]] · [[Machine Learning Specialization]]
