---
tags: [programming, python, tooling, notebook, data-science]
status: growing
level: 4
tier: 2
---

# Jupyter Notebook

A **Jupyter Notebook** is a document that mixes live code, its results (numbers, tables, charts) and your own written notes, in one page you can run piece by piece.

## 🟢 1. Simple (age 12)
A Jupyter Notebook is like a science lab notebook that can run experiments by itself. You write a little code in a box, press play, and the answer or the picture appears right underneath. Then you write a note about it, and go on to the next box.

**Example:** you load a list of exam scores in the first box, calculate the average in the second box, and draw a bar chart in the third. The chart sits right there under the code that made it, with your notes between the boxes.

## 🔵 2. How it works
1. A notebook is made of **cells**. A cell is either **code** or **text** (notes, headings, even maths).
2. You write [[Python]] in a code cell and run it (Shift + Enter). The result appears just below the cell.
3. Behind the page, a program called the **kernel** runs your code and **remembers everything** that you have run: every variable, every imported library ([[NumPy]], [[Pandas]]).
4. Charts from [[Matplotlib]] appear inline, right under the cell that drew them.
5. The numbers next to the cells, `In [1]`, `In [2]`, show the order in which you ran them, not the order in which they are written on the page.
6. You can run cells again, change them, and run them in any order. That is the strength and also the trap.
7. The notebook file keeps your code, your notes **and** the saved results together.
8. **The catch:** because the kernel remembers, a notebook can give a different answer from the same text depending on what you clicked earlier. Before you share it, restart the kernel and run all cells from the top, to see if it still works.

![[jupyter-notebook-screenshot.png]]
*A Jupyter code cell. The label `In [1]:` shows that this cell was run first, the code is in the grey box, and its output ("the free encyclopedia that anyone can edit") appears just below it. Underneath is an empty cell, `In [ ]:`, waiting for code. Image: Daniel Mietchen, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Screenshot_of_Jupyter_notebook_on_PAWS_with_SPARQL_query_for_motto_of_the_English_Wikipedia_%28cropped%29.png), CC BY-SA 4.0. I read the labels and the output text off the picture, not from the file text.*

## 🟠 3. Technical detail
- **File format:** a notebook is a plain text JSON file with the extension `.ipynb`, with the top-level keys `cells`, `metadata`, `nbformat` and `nbformat_minor` (the code below lists them). Each cell has its type, its source, its outputs and its execution count.
- **Kernel:** a separate process that runs the code. The notebook page talks to it. Restarting the kernel clears its memory but keeps the notebook file.
- **Hidden state:** the kernel keeps all variables, in the order they were run. If you run a cell twice, or out of order, the state changes. The code below shows `10` after running the cells once each and `100` after running the middle cell twice, from the same text.
- **Outputs are saved in the file,** so the file grows with pictures and long printed results. In the code below, a cell that prints 3,000 numbers makes the file 19,095 bytes, against 2,011 bytes once the outputs are cleared. That also makes the file noisy to compare in [[Git]].
- **Good habits:** run top to bottom before sharing, keep real, reusable code in `.py` files that you test with [[pytest]], and keep notebooks for exploring and explaining.
- **Environment:** a notebook uses the Python of its kernel, so install the kernel inside the project's environment ([[Virtual Environments]]).
- **Uses in machine learning:** trying ideas on a small data set, plotting results, and writing up experiments ([[PyTorch]], [[TensorFlow]], [[Evaluation Metrics]]).

```python
import json, nbformat
from nbclient import NotebookClient
cells = ["x = 1", "x = x * 10", "print(x)"]
def build(): 
    nb = nbformat.v4.new_notebook(); nb.cells = [nbformat.v4.new_code_cell(c) for c in cells]; return nb
def run(order):
    nb = build(); client = NotebookClient(nb, timeout=60, kernel_name="python3")
    with client.setup_kernel():
        for i in order: client.execute_cell(nb.cells[i], i)
    return nb.cells[2].outputs[0]["text"].strip()
print("cells:", cells)
print("Run All (cells 1, 2, 3 once each) prints:        ", run([0, 1, 2]))
print("cell 2 clicked twice by accident, then cell 3 prints:", run([0, 1, 1, 2]), "  <- same notebook text, different answer: the kernel remembers")
nb = build(); raw = json.loads(nbformat.writes(nb))
print("a notebook file is plain JSON with keys:", sorted(raw), "; each cell has", sorted(raw["cells"][0]))
loud = build(); loud.cells.append(nbformat.v4.new_code_cell("print(list(range(3000)))")); NotebookClient(loud, timeout=60, kernel_name="python3").execute()
with_out = len(nbformat.writes(loud)); 
for c in loud.cells: c.outputs = []; c.execution_count = None
print(f"file size with the outputs saved: {with_out:,} bytes; after clearing outputs: {len(nbformat.writes(loud)):,} bytes")
```

## 🔴 4. Deep / formal
**Execution model.** A notebook is a **read-eval-print loop** with a memory: each cell sends source code to the kernel, the kernel runs it against its shared global state, and prints the result back. The result of a cell is therefore a function of the cell **and of the whole history of cells run before it**, not of the page alone.

**Consequences:**
- **Reproducibility** needs a defined order. "Restart and run all" gives the order written on the page.
- **Deleted cells can still have effects:** a variable made by a cell you later deleted stays in the kernel until it restarts.
- **Large outputs** and embedded images are stored in the file, so version control sees them as changes.

**Trade-offs:**
- **Notebooks are good for exploring and explaining,** because you see each step's result at once.
- **They are weaker for building programs,** because of the hidden state, awkward testing and messy diffs. A common practice is to move finished code into modules (from memory).

**What came before, and how this improved on it:**
- **Before:** write a script, run it from the [[Command Line]] and read printed text, with charts opening in separate windows. Or the interactive Python prompt, which forgot everything when closed and could not mix in notes.
- **Its limit:** results, plots and your explanation lived in different places, and each change meant running the whole script again.
- **What changed:** one document where code, results and notes sit together and cells run one at a time against a remembering kernel. The notebook idea grew out of IPython and became Project Jupyter in 2014 (from memory, not checked; the earlier name was the IPython Notebook).
- **How much better:** the gain is working speed and clarity, which I cannot measure, so I give no figure for it. What the code below measures is the cost: the same three cells give `10` when run once each and `100` when the middle cell is run twice (hidden state), and saving the outputs of one 3,000-number cell takes the file from 2,011 bytes to 19,095 bytes.

**Check it yourself:** run the code in level 3 above (it starts a kernel and takes about 15 seconds; it needs `nbformat`, `nbclient` and `ipykernel`, which come with a Jupyter install). It prints every number quoted in this block.

## Related
- [[Python]] · [[NumPy]] · [[Pandas]] · [[Matplotlib]] · [[pytest]] · [[Git]] · [[Virtual Environments]] · [[Command Line]] · [[PyTorch]] · [[TensorFlow]] · [[Evaluation Metrics]] · [[Streamlit]]

## Sources
- Project Jupyter, https://jupyter.org (the `.ipynb` format, kernels and the history from IPython: from memory, not reopened this session).
- The code and the numbers were run for this note with nbformat 5.10.4, nbclient 0.10.0 and ipykernel 6.16.0.
- Image: [Screenshot of Jupyter notebook on PAWS with SPARQL query for motto of the English Wikipedia (cropped)](https://commons.wikimedia.org/wiki/File:Screenshot_of_Jupyter_notebook_on_PAWS_with_SPARQL_query_for_motto_of_the_English_Wikipedia_%28cropped%29.png) by Daniel Mietchen, CC BY-SA 4.0.
- Names and dates marked "from memory" are not checked.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dev-tools Jupyter Notebook:::A lab notebook that can run experiments by itself. You write a little code in a cell, run it, and the answer or the picture appears right underneath.

#flashcards/programming/concept #flashcards/topic/dev-tools What kinds of cell are there?::Code cells and text cells (notes, headings, even maths).

#flashcards/programming/concept #flashcards/topic/dev-tools What is the kernel?::A separate program that runs your code and remembers everything you ran: every variable and imported library. Restarting it clears its memory but keeps the notebook file.

#flashcards/programming/concept #flashcards/topic/dev-tools How do you run a cell?::Shift + Enter. The result appears just below the cell.

#flashcards/programming/example #flashcards/topic/dev-tools What do `In [1]`, `In [2]` show?::The order in which cells were run, not the order they are written on the page.

#flashcards/programming/concept #flashcards/topic/dev-tools What is the strength and the trap of running cells in any order?::You can rerun and change cells freely, but the kernel's hidden state changes. The note's code showed `10` after running cells once each and `100` after running many times.

#flashcards/programming/concept #flashcards/topic/dev-tools What should you do before sharing a notebook?::Restart the kernel and run all cells from the top, to see whether it still works.

#flashcards/programming/concept #flashcards/topic/dev-tools What is a `.ipynb` file?::A plain text JSON file with the top-level keys `cells`, `metadata`, `nbformat` and `nbformat_minor`.

#flashcards/programming/concept #flashcards/topic/dev-tools Why do notebook files grow large?::Outputs are saved in the file. In the note's code a cell printing 3,000 numbers made the file 19,095 bytes, against 2,011 bytes once outputs were cleared.

#flashcards/programming/concept #flashcards/topic/dev-tools Where should real reusable code live?::In `.py` files tested with pytest. Keep notebooks for exploring and explaining. Install the kernel inside the project's virtual environment.

#flashcards/programming/image #flashcards/topic/dev-tools ![[jupyter-notebook-screenshot.png]]
What does this picture show?
?
A Jupyter code cell. The label `In [1]:` shows this cell was run first, the code is in the grey box, and its output ("the free encyclopedia that anyone can edit") appears just below it. Underneath is an empty cell, `In [ ]:`, waiting for code. (From the note's caption: Daniel Mietchen, Wikimedia Commons.)
