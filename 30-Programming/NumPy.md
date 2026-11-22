---
tags: [numpy, python, programming, linear-algebra, machine-learning]
status: growing
level: 4
tier: 2
---

# NumPy

## 🟢 1. Simple (age 12)
NumPy is a **toolbox for [[Python]] that does maths on whole grids of numbers at once**. Instead of adding two long lists of numbers one pair at a time, you say "add these two lists" and it does all the pairs in one go, very fast.

**Example:** A teacher has 30 test marks and wants to add 5 bonus marks to everyone. Doing it by hand means 30 separate additions. With NumPy you write "marks plus 5" and all 30 change together. Machine learning is mostly this kind of bulk number crunching, which is why NumPy is everywhere.

## 🔵 2. How it works
1. NumPy's main object is the **array**: a grid of numbers that all have the same type. A row of numbers is a [[Vector]]. A table of rows and columns is a [[Matrix]]. Grids with more directions are called tensors.
2. Every array has a **shape** (how many rows, columns, and so on) and a **data type** (for example whole numbers or decimals).
3. You do maths on the **whole array** with one line: `a + b`, `a * 2`, `a.mean()`. No loop is needed.
4. **Broadcasting:** NumPy can combine arrays of different shapes when the shapes fit together. For example, subtract a row of averages from every row of a table in one step.
5. You can **slice** (pick out) parts of an array, and **reshape** it into a different grid with the same numbers.
6. Under the hood, the heavy loops run in fast compiled code, not in slow Python. [[Neural Network]] libraries such as PyTorch work in the same style, but run on a [[GPU]].

![[numpy-array-structure.webp]]
*How NumPy stores a grid (left: the 4 by 3 grid `x`). The numbers actually sit in one long flat row in memory (top right). The array object only keeps the recipe for reading it: where the data starts, the data type (8-byte integers), the **shape** $(4,3)$, and the **strides** $(24,8)$, which say how many bytes to jump to go one row down (24) or one column right (8). Image: Harris et al., from "Array programming with NumPy" (Nature, 2020), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:The_NumPy_array_data_structure_and_its_associated_metadata_fields.webp), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Create:** `np.array([...])`, `np.zeros((3, 4))`, `np.arange(12)`, `np.linspace(0, 1, 5)`, `np.random.default_rng(0).normal(size=...)`.
- **Shape tools:** `.shape`, `.reshape(rows, cols)`, `.T` (transpose, see [[Matrix]]), `np.newaxis` to add a direction.
- **Maths over arrays:** `+ - * /` act **element by element**. `@` is real [[Matrix]] multiplication.
- **Reductions:** `sum`, `mean`, `max`, `argmax` with an `axis` choosing which direction to collapse.
- **Broadcasting rule:** compare shapes from the right; two sizes are compatible if they are equal or one of them is 1. A size of 1 is stretched to match.
- **Slicing and masks:** `a[1:3, :]`, `a[a > 0]`.
- **Views vs copies:** slicing usually gives a **view** that shares memory with the original (changing one changes the other). `.copy()` makes a separate one.
- **Data types matter:** `float32` uses half the memory of `float64`. Deep learning often uses small types (see [[GPU]]). The default integer size depends on the platform and NumPy version, so state `dtype` if it matters.
- **Why it is fast:** one Python call runs a tight compiled loop over contiguous memory. Prefer array expressions over Python `for` loops.

```python
import numpy as np
x = np.arange(12, dtype=np.int64).reshape(4, 3)
print(x.shape, x.strides, x.dtype)       # (4, 3) (24, 8) int64
print(x[2, 1])                            # 7

marks = np.array([[60., 70.], [80., 90.]])
print(marks + 5)                          # every mark +5
print(marks - marks.mean(axis=0))         # broadcasting: subtract each column's mean
# [[-10. -10.]
#  [ 10.  10.]]

A = np.array([[1., 2.], [3., 4.]]); B = np.array([[5., 6.], [7., 8.]])
print(A * B)                              # element by element
print(A @ B)                              # matrix multiplication
# [[19. 22.]
#  [43. 50.]]
```

## 🔴 4. Deep / formal
**How an array finds an element: strides.** An array is one flat block of memory plus a recipe. The position of the element at index $(i_1,i_2,\dots,i_k)$ is:

$$
\text{offset}=\sum_{d=1}^{k} i_d\,s_d \quad\text{(in bytes, counted from the start of the data)}
$$

**Where:**
- $k$ is the number of directions (dimensions) of the array: 1 for a vector, 2 for a grid.
- $i_d$ is your index along direction $d$ (which row, which column, and so on).
- $s_d$ is the **stride** of direction $d$: how many bytes to jump to move one step along that direction. **Matters:** it is what lets NumPy reshape or transpose an array without copying any data. It only changes the strides.
- $\text{offset}$ is how far from the start of the data the element lives. Dividing by the bytes per element gives its place in the flat row.
- $\sum_d$ adds up the jumps from every direction.

**Example (tiny numbers):** the picture's array: shape $(4,3)$, 8-byte integers, strides $(24,8)$. Find the element at row 2, column 1.
- Row jump: $2\cdot24=48$ bytes. Column jump: $1\cdot8=8$ bytes.
- Offset $=48+8=\mathbf{56}$ bytes.
- Bytes per element is 8, so position $=56/8=\mathbf{7}$ in the flat row, which holds the number $7$. That matches `x[2, 1]` in the code.
- Meaning: the transpose $x^\top$ has shape $(3,4)$ and strides $(8,24)$: the same 12 numbers, just read with the jumps swapped. Nothing was moved.

**Further notes:**
- **Memory layout:** the default is row-major ("C order": the last index changes fastest). Reading along rows is therefore fast and reading down columns jumps through memory, which can be slower for big arrays.
- **Vectorisation and the CPU:** array operations can use SIMD instructions (see [[GPU]], where the same idea is scaled up thousands of times).
- **Broadcasting does not copy:** the stretched dimension is given a stride of 0, so the same numbers are re-read. Memory stays small.
- **Floating-point caveat:** sums of decimals are only approximately exact (for example $0.1+0.2\neq0.3$ exactly), so compare with `np.isclose`, not `==`.
- **Relation to other tools:** PyTorch tensors and JAX arrays copy NumPy's ideas (shape, dtype, broadcasting, strides) and add GPU support and automatic gradients for [[Backpropagation]].

**What came before, and how this improved on it:**
- **Before:** Python lists and loops.
- **Its limit:** each Python integer is a full object, and each loop step runs through the interpreter.
- **What changed:** one block of same-type numbers with operations that run in compiled code. NumPy was created in 2005 by Travis Oliphant by merging Numeric and Numarray (from memory, not checked).
- **How much better:** in the code below, 1,000,000 integers take 36,000,052 bytes as a Python list and 8,000,000 as a NumPy array, 4.5 times smaller. A sum of squares took about 21 to 35 times less time in three runs here, and the answers matched. The timing depends on the machine. **Catch:** the first version of this snippet gave a wrong sum, because NumPy's default integer was 32 bits here and overflowed; the code now asks for 64 bits.

**Check it yourself:**

```python
import sys, numpy as np, time
n = 1_000_000
lst = list(range(n)); arr = np.arange(n, dtype=np.int64)
mem_list = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst); mem_arr = arr.nbytes
print(f"memory for {n:,} integers: Python list {mem_list:,} bytes, NumPy array {mem_arr:,} bytes ({mem_list / mem_arr:.1f}x smaller)")
t0 = time.perf_counter(); s1 = sum(x * x for x in lst); t1 = time.perf_counter(); s2 = int((arr * arr).sum()); t2 = time.perf_counter()
print("same sum of squares:", s1 == s2, f"  loop {t1 - t0:.3f} s, array {t2 - t1:.5f} s, ratio about {(t1 - t0) / max(t2 - t1, 1e-9):,.0f}x (depends on this machine)")
```

## Related
- [[Python]] · [[Vector]] · [[Matrix]] · [[Neural Network]] · [[GPU]] · [[Backpropagation]] · [[Normal Distribution]] · [[Pandas]] · [[Matplotlib]] · [[PyTorch]] · [[Big-O]] · [[Arrays and Lists]] · [[TensorFlow]] · [[Command Line]] · [[Virtual Environments]] · [[pytest]] · [[Jupyter Notebook]] · [[Vectorization]]

## Sources
- Harris et al., *Array programming with NumPy*, Nature, 2020
- NumPy documentation: *NumPy: the absolute basics for beginners* and the broadcasting guide (numpy.org)

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/data-libs NumPy:::A Python toolbox that does maths on whole grids of numbers at once, instead of one pair at a time.

#flashcards/programming/concept #flashcards/topic/data-libs What is a NumPy array?::A grid of numbers that all have the same type. A row is a vector, a table is a matrix, and grids with more directions are tensors.

#flashcards/programming/concept #flashcards/topic/data-libs What do shape and data type describe?::Shape is how many rows, columns and so on. Data type is, for example, whole numbers or decimals.

#flashcards/programming/concept #flashcards/topic/data-libs How do you do maths on the whole array?::With one line and no loop: `a + b`, `a * 2`, `a.mean()`.

#flashcards/programming/reverse #flashcards/topic/data-libs Broadcasting:::NumPy combines arrays of different shapes when the shapes fit together, for example subtracting a row of averages from every row of a table in one step.

#flashcards/programming/concept #flashcards/topic/data-libs Slice and reshape?::Slicing picks out parts of an array. Reshaping turns it into a different grid with the same numbers.

#flashcards/programming/concept #flashcards/topic/data-libs Why is NumPy fast?::The heavy loops run in fast compiled code, not in slow Python.

#flashcards/programming/formula #flashcards/topic/data-libs Write the byte position of an array element and explain each symbol.
?
$$\text{offset}=\sum_{d=1}^{k} i_d\,s_d \quad\text{(in bytes, counted from the start of the data)}$$
- $k$: the number of directions (dimensions) of the array. 1 for a vector, 2 for a grid.
- $i_d$: your index along direction $d$ (which row, which column).
- $s_d$: the stride of direction $d$, how many bytes to jump to move one step along it. This is what lets NumPy reshape or transpose an array without copying any data. It only changes the strides.
- $\text{offset}$: how far from the start of the data the element sits.

#flashcards/programming/example #flashcards/topic/data-libs Shape $(4,3)$, 8-byte integers, strides $(24,8)$. Where is the element at row 2, column 1? And the transpose?::Row jump $2\cdot24=48$, column jump $1\cdot8=8$, offset $56$ bytes, so position $56/8=7$ in the flat row, which holds the number 7 (matches `x[2, 1]`). The transpose has shape $(3,4)$ and strides $(8,24)$: the same 12 numbers, only the strides change.

#flashcards/programming/image #flashcards/topic/data-libs ![[numpy-array-structure.webp]]
What does this picture show?
?
How NumPy stores a grid (left: the 4 by 3 grid `x`). The numbers sit in one long flat row in memory (top right). The array object keeps only the recipe for reading it: where the data starts, the data type (8-byte integers), the shape $(4,3)$, and the strides $(24,8)$, how many bytes to jump to go one row down (24) or one column right (8). (From the note's caption: Harris et al., "Array programming with NumPy", Nature 2020, Wikimedia Commons.)
