---
tags: [python, programming, tooling, machine-learning]
status: growing
level: 4
tier: 1
---

# Python

## 🟢 1. Simple (age 12)
Python is a **language for giving a computer instructions** in words that are close to plain English. You write a few lines, press run, and the computer does what you said. It is popular because it is easy to read, and almost all AI work is done with it.

**Example:** Telling a robot how to make a sandwich. In Python you might write `bread = 2`, `cheese = 1`, `print(bread + cheese)`. The computer reads each line from top to bottom, one step at a time, and prints `3`.

## 🔵 2. How it works
1. You write instructions in a text file (a **script**), for example `hello.py`.
2. A program called the **interpreter** reads the file and runs it line by line, top to bottom.
3. You store things in **variables** (labelled boxes): numbers, text, lists.
4. You group steps you want to reuse into a **function** (`def`) and call it by name.
5. You repeat steps with a **loop** (`for`) and make choices with an **if**.
6. You borrow tools other people wrote with `import`. For AI work these are libraries such as NumPy (fast number crunching with [[Vector]]s and [[Matrix]] grids) and PyTorch (building a [[Neural Network]] and running it on a [[GPU]]).

Python also cares about **indentation**: the spaces at the start of a line show which lines belong together. That is how the computer knows where a loop or function ends.

![[python-parse-tree.png]]
*What the interpreter does with your text: it first cuts the code into pieces (the grey "Tokenization" box, the same idea as [[Tokenization]] for language models), then arranges the pieces into a tree that shows their structure. This is the tree for the one-line function `def add5(x): return x + 5`. Image: Lulu of the Lotus-Eaters (English Wikipedia), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Python_add5_parse.png), public domain.*

## 🟠 3. Technical detail
- **Types you will meet:** numbers (`int`, `float`), text (`str`), `list` (ordered, changeable), `tuple` (ordered, fixed), `dict` (label to value), `set` (unique items).
- **Comprehensions:** one-line loops that build a list: `[n * n for n in nums]`.
- **Functions** are values too: you can pass them around, which is how libraries accept your code.
- **Running a script:** `python hello.py`. Always run Python code you did not write only after reading it.
- **Virtual environments:** a separate folder of installed libraries per project (`python -m venv .venv`), so projects do not clash.
- **Packages:** installed with `pip install <name>`. Pin versions in a `requirements.txt` for repeatable work.
- **Speed:** plain Python is slow for heavy number work, so libraries run the heavy loops in fast compiled code (NumPy, PyTorch). Write whole-array operations instead of Python loops.
- **Notebooks:** Jupyter notebooks mix code, results and notes, and are common for experiments.

```python
def add5(x):
    return x + 5
print(add5(2))                          # 7

nums = [3, 1, 2]
print(sorted(nums), nums)               # [1, 2, 3] [3, 1, 2]   sorted() makes a new list
print([n * n for n in nums])            # [9, 1, 4]

counts = {}
for w in "the cat and the hat".split():
    counts[w] = counts.get(w, 0) + 1
print(counts)                           # {'the': 2, 'cat': 1, 'and': 1, 'hat': 1}
```

## 🔴 4. Deep / formal
**How code actually runs (CPython, the usual interpreter):**
1. **Tokenise and parse** the text into a syntax tree (the picture above).
2. **Compile** the tree to **bytecode**: a list of small simple instructions for a virtual machine. You can look at it with the standard `dis` module (`import dis; dis.dis(add5)`). The exact instructions change between Python versions, so I have not copied any here.
3. **Execute** the bytecode in the Python virtual machine, one instruction at a time. That extra layer is a big reason plain Python is slower than compiled languages.

**Variables are labels, not boxes.** A variable name points at an object. Assigning `b = a` makes a second label on the **same** object. This matters for **mutable** objects (lists, dicts), which can be changed in place. Numbers, strings and tuples are **immutable**: operations create new ones.

```python
a = [1, 2]
b = a                 # same list, two labels
b.append(3)
print(a)              # [1, 2, 3]   a changed too!
c = a.copy()          # a new, separate list
c.append(4)
print(a, c)           # [1, 2, 3] [1, 2, 3, 4]

def bad(x, acc=[]):   # the default list is made ONCE, when the def runs
    acc.append(x)
    return acc
print(bad(1), bad(2)) # [1, 2] [1, 2]  both calls share the same list
```

**Why the last result surprises people:** the default value `[]` is created once and reused by every call. The usual fix is `acc=None` and then `if acc is None: acc = []` inside.

**Other deep points:**
- **Everything is an object**, including functions and classes. Names are looked up in nested scopes (local, enclosing, global, built-in).
- **The GIL (global interpreter lock)** in standard CPython lets only one thread run Python bytecode at a time, so threads do not speed up heavy computation. Use multiple processes, or libraries that release the lock. (Stated from memory; recent Python versions are adding an optional mode without it, so check the current docs.)
- **Garbage collection:** objects are freed automatically when nothing points to them (reference counting plus a cycle detector).
- **Type hints** (`def f(x: int) -> int`) are optional notes for humans and tools. Python does not enforce them when running.
- **Why Python dominates AI:** it reads like pseudo-code, has great libraries, and acts as the "glue" while GPUs do the heavy lifting inside the libraries ([[Backpropagation]], [[Gradient Descent]] and the [[Transformer]] all run in compiled code called from Python).

**What came before, and how this improved on it:**
- **Before:** languages such as C and Java, with fixed-size integers, declared types and more lines for the same job.
- **Its limit:** fixed-size integers wrap around silently, and the extra code makes experiments slower to write.
- **What changed:** a language built for readability, with integers of any size and short built-in data structures. Guido van Rossum started it in the late 1980s and released it in 1991 (from memory, not checked).
- **How much better:** in the code below, 25! is exact in Python (15511210043330985984000000) while 64-bit integers wrapped to 7034535277573963776 without any warning. A word count takes 3 lines. **Honest cost:** plain Python is much slower than C, which is why [[NumPy]] and other libraries run the heavy loops in compiled code.

**Check it yourself:**

```python
import numpy as np
import math
exact = math.factorial(25); fixed = np.int64(1)
with np.errstate(all="ignore"):
    for i in range(1, 26): fixed = np.int64(fixed * i)
print("25! in Python:      ", exact)
print("25! in 64-bit ints: ", int(fixed), " (wrapped around, silently wrong)")
text = "to be or not to be"
counts = {}
for w in text.split(): counts[w] = counts.get(w, 0) + 1
print("word count in 3 lines of Python:", counts)
```

## Related
- [[Vector]] · [[Matrix]] · [[Neural Network]] · [[GPU]] · [[Tokenization]] · [[LLM]] · [[Backpropagation]] · [[Pandas]] · [[Matplotlib]] · [[PyTorch]] · [[Big-O]] · [[Hash Tables]] · [[Binary Search]] · [[Sorting]] · [[Recursion]] · [[Dynamic Programming]] · [[Arrays and Lists]] · [[Stacks and Queues]] · [[Graphs]] · [[Heaps and Priority Queues]] · [[Trees]] · [[TensorFlow]] · [[Git]] · [[Command Line]] · [[Virtual Environments]] · [[SQL]] · [[Regular Expressions]] · [[pytest]] · [[Jupyter Notebook]] · [[FastAPI]] · [[Streamlit]] · [[Docker]]

## Sources
- The Python Tutorial, docs.python.org (free)
- Sweigart, *Automate the Boring Stuff with Python* (free online)
- Python docs for the `dis` module

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/python-core Python:::A language for giving a computer instructions in words close to plain English. Easy to read, and almost all AI work is done with it.

#flashcards/programming/concept #flashcards/topic/python-core What is the interpreter?::A program that reads your script file (for example `hello.py`) and runs it line by line, top to bottom.

#flashcards/programming/concept #flashcards/topic/python-core How does Python know where a loop or function ends?::By indentation: the spaces at the start of a line show which lines belong together.

#flashcards/programming/concept #flashcards/topic/python-core How do you group reusable steps, repeat steps, make choices, and borrow tools?::A function with `def`, a loop with `for`, a choice with `if`, and borrowed tools with `import`.

#flashcards/programming/concept #flashcards/topic/python-core Name the main Python types.::Numbers (`int`, `float`), text (`str`), `list` (ordered, changeable), `tuple` (ordered, fixed), `dict` (label to value) and `set` (unique items).

#flashcards/programming/concept #flashcards/topic/python-core What is a comprehension?::A one-line loop that builds a list, for example `[n * n for n in nums]`.

#flashcards/programming/concept #flashcards/topic/python-core How do you run a script?::`python hello.py`. Run Python code you did not write only after reading it.

#flashcards/programming/concept #flashcards/topic/python-core How do projects avoid clashing libraries?::A virtual environment per project (`python -m venv .venv`). Packages are installed with `pip install <name>`, and versions pinned in a `requirements.txt`.

#flashcards/programming/concept #flashcards/topic/python-core Why write whole-array operations instead of Python loops?::Plain Python is slow for heavy number work, so libraries (NumPy, PyTorch) run the heavy loops in fast compiled code.

#flashcards/programming/image #flashcards/topic/python-core ![[python-parse-tree.png]]
What does this picture show?
?
What the interpreter does with your text: it first cuts the code into pieces (the grey "Tokenization" box, the same idea as tokenization for language models), then arranges the pieces into a tree showing their structure. This is the tree for the one-line function `def add5(x): return x + 5`. (From the note's caption: Lulu of the Lotus-Eaters, Wikimedia Commons, public domain.)

#flashcards/programming/cloze #flashcards/topic/python-core Python functions are ==values too==: you can pass them around, which is how libraries accept your code.
