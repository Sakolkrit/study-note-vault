---
tags: [algorithms, data-structures, arrays, linked-list, programming, python]
status: growing
level: 4
tier: 1
---

# Arrays and Lists

An **array** is a row of boxes of the same size, stored one after another in the computer's memory, and numbered from 0. It is the simplest and most important way to store a collection of things. The thing Python calls a `list` is a *dynamic array*: an array that can grow. A **linked list** is the main alternative: items are scattered around and each one points to the next. Which one you choose changes how fast different jobs are (see [[Big-O]]), and many other structures, such as the [[Hash Tables]] and the sorted arrays used in [[Binary Search]] and [[Sorting]], are built on arrays.

## 🟢 1. Simple (age 12)
Picture a row of lockers in a school corridor, all the same size, numbered 0, 1, 2, 3 and so on. If someone says "get what is in locker 7", you walk straight to locker 7. You do not open the lockers before it. That is an array: **jumping to any item is instant, because its number tells you where it is.** But if you want to squeeze a new locker in at the very start, every other locker has to move one place along, and that is a lot of work.

Now picture a treasure hunt instead. Each clue tells you where the next clue is hidden. Adding a new clue at the start is easy: write it and say "the next one is the old first clue". But to get clue number 7 you must follow the chain from the beginning, clue after clue. That is a linked list.

**Example:** A queue of children waiting for a bus, standing on numbered spots, is an array: "the child on spot 4" is easy to find. If a new child pushes in at the front, everyone behind must shuffle back a place.

## 🔵 2. How it works
1. **An array reserves one block of memory.** All items are the same size, so item number $i$ is found by adding $i\times$(item size) to the start of the block. That is one calculation, however long the array is. This is why reading `a[i]` is fast.
2. **Adding at the end is cheap.** A Python list keeps some **spare room** after its last item. `append` just fills the next free slot.
3. **When the spare room runs out, the list moves house.** It asks for a bigger block (with more spare room), copies every item across, and carries on. This copy is slow, but it happens rarely, so on average each `append` is still cheap (an *amortised* cost, see [[Big-O]]).
4. **Adding at the front (or in the middle) is expensive.** Every item after the new one must shift along by one place to make room.
5. **A linked list does the opposite.** Each item is a small box holding a value and the address of the next box. Adding at the front is one step. But to reach item number $i$, you must follow $i$ links from the start.

![[dynamic-array.svg]]
*A dynamic array growing as numbers are added. The black boxes hold data (the "logical size"). The dotted boxes are spare room. The whole row, black and dotted, is the "capacity". When the spare room runs out (the turtles mark the slow steps, with a capacity of 2 and then 4), the array moves to a block twice as big. Image: Derrick Coetzee (Dcoetzee), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dynamic_array.svg), CC0.*

![[singly-linked-list.svg]]
*A singly linked list with three numbers (12, 99, 37). Each box holds a value and an arrow to the next box. The last box points to nothing (the crossed box). Image: Lasindi (vectorisation), based on an image by Derrick Coetzee, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Singly-linked-list.svg), public domain.*

## 🟠 3. Technical detail
- **What a Python `list` really is:** a dynamic array of *pointers* to objects, not of the numbers themselves. That is why one list can hold numbers, text and other lists. The price is extra memory and a little more work than a plain array of numbers. Libraries such as [[NumPy]] store the numbers themselves side by side, which is much more compact and fast (and the reason it is used for [[Matrix]] work).
- **Costs of a Python list** (see [[Big-O]]):
  - `a[i]`, `len(a)`: $O(1)$.
  - `append`, `pop()` at the end: $O(1)$ amortised.
  - `insert(0, x)`, `pop(0)`, and removing from the middle: $O(n)$, because items shift.
  - `x in a` and `a.index(x)`: $O(n)$, a scan. A set or dictionary is faster for this (see [[Hash Tables]]).
  - Slicing `a[i:j]`: $O(j-i)$, because it copies.
- **Growth is not exactly doubling:** the picture shows doubling, but the code shows what Python 3.9 does: spare room is reserved in steps of about one eighth of the length, after a small start (4, 8, 16, 24, 32, 40 slots at lengths 1, 5, 9, 17, 25, 33). The exact rule can differ between Python versions (not checked on other versions). Any growth by a fixed *factor* gives $O(1)$ amortised appends (equation 2 below).
- **Linked list:** cheap insert or delete at a place you already hold, at the cost of slow reading by position and extra memory for the pointers. It is rarely better in Python. On real computers, arrays are also friendlier to the cache, which makes them faster than the operation counts alone suggest (from memory; I did not measure it).
- **`collections.deque`:** a double-ended queue in Python's standard library. Adding or removing at *either* end is $O(1)$. Use it when you need a queue. Reading from the middle is slower than a list (from memory, I did not test it here).
- **Common traps:** (1) a slice copies the outer list only, so inner lists are still shared; (2) `[[0]*3]*3` repeats the *same* inner list three times, so changing one row changes all (the code shows both); (3) building a long text with repeated `+=` can copy a lot, so use `"".join(parts)` (from memory, not tested here).
- **Limits:** the code counts operations (moves, copies, hops). It does not time anything, and I did not compare with a C array or other languages.

The code covers six things: (1) the address formula, checked by reading real memory, (2) the spare room a Python list keeps, (3) how many items are copied while growing, (4) the cost of inserting at the front, (5) a linked list's reading cost, (6) `deque` and the two traps.

```python
import sys, array, ctypes
from collections import deque
# 1) an array stores same-size items side by side, so item i sits at base + i * size: one multiplication, one jump
a = array.array('i', [10, 20, 30, 40, 50]); base, n = a.buffer_info(); size = a.itemsize
print('item size', size, 'bytes | length', n, '| reading memory at base + 3*size gives', ctypes.c_int.from_address(base + 3 * size).value, '(item 3 is', a[3], ')')
# 2) a Python list holds pointers; it keeps spare room, so append is usually cheap
l, caps, last = [], [], None
for i in range(40):
    l.append(i); s = sys.getsizeof(l)
    if s != last: caps.append((len(l), (s - sys.getsizeof([])) // 8)); last = s
print('(length, slots reserved) each time the list grew:', caps)
# 3) own dynamic array with a growth factor g, counting every item copied during a resize
def append_cost(n, g):
    cap, size, copies = 1, 0, 0
    for _ in range(n):
        if size == cap: copies += size; cap = int(cap * g) + 1 if g < 2 else cap * 2    # copy all items to a bigger block
        size += 1
    return copies
for n in (8, 1000, 1000000): print('n', n, '| g=2 copies', append_cost(n, 2), '| bound g/(g-1)*n =', 2 * n)
print('g=1.125 (CPython-like), n=1000000: copies', append_cost(1000000, 1.125), '| bound g/(g-1)*n =', round(1.125 / 0.125 * 1000000))
# 4) inserting at the front of an array shifts everything: count the moves
def front_moves(n):
    moves = 0
    for size in range(n): moves += size                                       # every existing item moves one place right
    return moves
for n in (10, 1000): print('n', n, 'front inserts: moves', front_moves(n), '| n(n-1)/2 =', n * (n - 1) // 2, '| n appends: at most', 2 * n, 'copies')
# 5) a linked list: front insert is one step, but reaching item i means following i links
class Node:
    def __init__(self, v, nxt=None): self.v, self.next = v, nxt
head = None
for v in range(1000): head = Node(v, head)                                      # 1000 front inserts, one new node each
def get(head, i):
    hops = 0
    while hops < i: head = head.next; hops += 1
    return head.v, hops
print('item 0 of the linked list:', get(head, 0), '| item 999:', get(head, 999), '(hops = index)')
# 6) deque (a chain of blocks) allows cheap ends; list.pop(0) is the slow end
d = deque(range(5)); d.appendleft(-1); d.append(5)
print('deque after appendleft and append:', list(d), '| popleft gives', d.popleft())
# 7) traps: a slice is a copy (of the pointers), and * repeats the same inner list
x = [[1], [2]]; y = x[:]; y[0].append(9)
print('slice copies the outer list:', y is not x, '| but inner lists are shared:', x[0], y[0], x[0] is y[0])
g = [[0] * 3] * 3; g[0][0] = 7
h = [[0] * 3 for _ in range(3)]; h[0][0] = 7
print('[[0]*3]*3 ->', g, '| comprehension ->', h)
# item size 4 bytes | length 5 | reading memory at base + 3*size gives 40 (item 3 is 40 )
# (length, slots reserved) each time the list grew: [(1, 4), (5, 8), (9, 16), (17, 24), (25, 32), (33, 40)]
# n 8 | g=2 copies 7 | bound g/(g-1)*n = 16
# n 1000 | g=2 copies 1023 | bound g/(g-1)*n = 2000
# n 1000000 | g=2 copies 1048575 | bound g/(g-1)*n = 2000000
# g=1.125 (CPython-like), n=1000000: copies 8680621 | bound g/(g-1)*n = 9000000
# n 10 front inserts: moves 45 | n(n-1)/2 = 45 | n appends: at most 20 copies
# n 1000 front inserts: moves 499500 | n(n-1)/2 = 499500 | n appends: at most 2000 copies
# item 0 of the linked list: (999, 0) | item 999: (0, 999) (hops = index)
# deque after appendleft and append: [-1, 0, 1, 2, 3, 4, 5] | popleft gives -1
# slice copies the outer list: True | but inner lists are shared: [1, 9] [1, 9] True
# [[0]*3]*3 -> [[7, 0, 0], [7, 0, 0], [7, 0, 0]] | comprehension -> [[7, 0, 0], [0, 0, 0], [0, 0, 0]]
```
Reading the output (Python 3.9, standard library only):
- **Address:** an array of 5 four-byte integers is read at `base + 3*4`, and the number found is 40, which is item 3.
- **Spare room:** the list grew at lengths 1, 5, 9, 17, 25 and 33, reserving 4, 8, 16, 24, 32 and 40 slots. So 40 appends caused only 6 resizes.
- **Growing cost:** with doubling, 8 appends copied 7 items, 1,000 appends copied 1,023, and 1,000,000 appends copied 1,048,575 ($2^{20}-1$). All are below $2n$. With a factor of about 1.125 (my simulation of Python's rule, with a rounding shortcut), a million appends copied 8,680,621, below the bound of $9n=9{,}000{,}000$.
- **Front inserts:** 10 inserts at the front moved 45 items and 1,000 moved 499,500, exactly $n(n-1)/2$. The same number of appends needs at most $2n$ copies.
- **Linked list:** item 0 takes 0 hops, item 999 takes 999 hops.
- **Deque:** adding at both ends and removing from the left behave as expected.
- **Traps:** the slice is a different list but the inner list is the same object (the append through `y` shows in `x`). `[[0]*3]*3` changes all three rows, the comprehension changes one.
- **Limits:** only counts, no timing, small lists.

## 🔴 4. Deep / formal

**Finding an item.** In an array of items that each take $s$ bytes, starting at memory address $b$, item number $i$ (counting from 0) is at:

$$\text{address}(i)=b+i\cdot s$$

**Where:**
- $b$ is the base: the address where the array starts. **Matters:** it is the only thing the computer must remember about the array.
- $i$ is the index (0 for the first item). **Matters:** counting from 0 is what makes the first item sit exactly at $b$. Counting from 1 would need an extra subtraction.
- $s$ is the size of one item in bytes. **Matters:** items must all be the same size, otherwise the multiplication gives the wrong place. This is why Python's list stores equal-sized pointers, not the objects.
- The result is one multiplication and one addition, whatever the value of $i$. **Matters:** that is the $O(1)$ reading cost.

**Example (tiny numbers):** $b=1000$, $s=4$, $i=3$.
1. $\text{address}(3)=1000+3\times4=1012$.
2. In the code the same formula, applied to the real address of an integer array, read the number 40, which is item 3 of the array.
3. Meaning: the address comes from a calculation, not from walking past items 0, 1 and 2. A linked list has no such formula and needs 3 hops.

**The cost of growing.** Say the array grows by a factor $g>1$ each time it is full (the picture has $g=2$). Over $n$ appends, the total number of items copied during resizes is at most:

$$\text{copies}\ \le\ n\left(1+\frac1g+\frac1{g^2}+\cdots\right)=\frac{g}{g-1}\,n$$

**Where:**
- $n$ is the number of appends. **Matters:** the total work is proportional to $n$, so each append costs a fixed amount on average: $O(1)$ amortised.
- $g$ is the growth factor (2 in the picture; about 1.125 in Python's rule). **Matters:** a bigger $g$ means fewer copies but more unused room. Without any growth factor (for example, adding only one slot at a time, $g\to1$) the bound blows up and the cost becomes $O(n^2)$ in total.
- The brackets are a geometric series. **Matters:** the last copy moves at most $n$ items, the one before at most $n/g$, the one before that at most $n/g^2$, and so on. The terms shrink fast, so the sum stays a fixed multiple of $n$.
- $g/(g-1)$ is the sum of that series. **Matters:** it is 2 for doubling, and 9 for $g=1.125$.

**Example (tiny numbers):** doubling ($g=2$), $n=8$ appends, starting from capacity 1.
1. The array is full after 1, 2 and 4 items, so it copies 1, then 2, then 4 items: $1+2+4=7$ copies.
2. The bound is $\frac{2}{1}\times8=16$. $7\le16$. The code counted 7.
3. For $n=1000$ the code counted 1,023 copies against a bound of 2,000.
4. Meaning: although one resize can be slow, the total copying is less than twice the number of appends, so each append costs a small fixed amount on average. This is what "amortised $O(1)$" means.

**The cost of inserting at the front.** Putting an item at the front of an array that already holds $k$ items shifts all $k$ of them. Doing this $n$ times in a row, starting from an empty array, moves:

$$\sum_{k=0}^{n-1}k=\frac{n(n-1)}{2}$$

**Where:**
- $k$ is how many items are in the array just before the insert. **Matters:** each of them has to shift one place, so the cost of one insert grows with the length.
- The sum adds the cost of each insert, from $k=0$ (empty array) to $k=n-1$. **Matters:** the costs add up to a total that grows like $n^2$.
- $\frac{n(n-1)}{2}$ is the sum of the numbers $0,1,\dots,n-1$ (the pairing trick: first plus last, second plus second-last, and so on, each pair adds to $n-1$, and there are $n/2$ pairs). **Matters:** it is $O(n^2)$, so doubling $n$ makes the work four times as large. A linked list or a `deque` makes the same job $O(n)$.

**Example (tiny numbers):** $n=4$.
1. Inserts happen into arrays holding 0, 1, 2 and 3 items. They move $0+1+2+3=6$ items.
2. The formula gives $4\times3/2=6$.
3. For $n=1000$ the code counted 499,500 moves, while 1,000 appends need at most 2,000 copies.
4. Meaning: front-inserting 1,000 items into a list takes about 250 times more item moves than appending them (499,500 against at most 2,000). The gap widens as $n$ grows. This is a count of moves, not a measured time.

**Other points (from memory, not checked here):**
- The "amortised" analysis can also be done by the accounting method (each append pays a few extra "coins" for later copying) or the potential method. I showed only the direct sum.
- Whether a deque or a list is faster for real jobs depends on the machine and Python version. Measure before deciding.
- A doubly linked list (each node also points back) makes deleting a node you already hold $O(1)$. I did not cover it.
- Arrays are the basis of the [[Matrix]] and tensor types in [[PyTorch]] and [[NumPy]], where the address formula extends to several indices (using "strides").

**What came before, and how this improved on it:**
- **Before:** a fixed-size block of memory, which must be made big enough in advance.
- **Its limit:** if you guess too small, you cannot add more, and if you guess too big, you waste memory.
- **What changed:** a dynamic array that grows by doubling when full (the way Python's `list` works, with a smaller growth factor in practice; this is from memory). A linked list is the other choice: no copying on growth, and fast insertion at the front, but slow to reach the middle.
- **How much better:** in the code below, 1,000,000 appends cost 499,999,500,000 item copies when growing by one slot each time, and 1,048,575 when doubling, which is 476,837 times fewer (about 1.05 copies per append). Inserting at the front of an array of 100,000 shifts 100,000 items, while a linked list changes one link.

**Check it yourself:**

```python
n = 1_000_000
# grow by one slot each time: every append copies the whole array
by_one = n * (n - 1) // 2
# double when full: copy only at sizes 1, 2, 4, ...
cap, size, doubling = 1, 0, 0
for _ in range(n):
    if size == cap: doubling += size; cap *= 2
    size += 1
print(f"{n:,} appends, counting items copied when the array has to grow:")
print(f"  grow by one slot: {by_one:,} copies")
print(f"  double when full: {doubling:,} copies ({by_one // doubling:,}x fewer, about {doubling / n:.2f} per append)")
m = 100_000
print(f"insert at the front of {m:,} items: array shifts {m:,} items, linked list rewires 1 link")
```

## Related
[[Big-O]] · [[Hash Tables]] · [[Binary Search]] · [[Sorting]] · [[Recursion]] · [[Dynamic Programming]] · [[NumPy]] · [[Pandas]] · [[Python]] · [[Matrix]] · [[Stacks and Queues]] · [[Graphs]] · [[Heaps and Priority Queues]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, on amortised analysis and dynamic tables (from memory; I did not open it for this note).
- Python documentation for `list`, `array` and `collections.deque` (docs.python.org). I did not read CPython's source: the growth steps in the code were observed by running Python 3.9.
- Figures: [Dynamic_array.svg](https://commons.wikimedia.org/wiki/File:Dynamic_array.svg) (Dcoetzee, CC0) and [Singly-linked-list.svg](https://commons.wikimedia.org/wiki/File:Singly-linked-list.svg) (Lasindi, based on an image by Derrick Coetzee, public domain), on Wikimedia Commons. I viewed both before writing the captions.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/data-structures Array:::A row of same-size lockers numbered from 0. Jumping to any item is instant, because its number tells you where it is.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the address of array item $i$ and explain each symbol.
?
$$\text{address}(i)=b+i\cdot s$$
- $b$: the base, the address where the array starts. It is the only thing the computer must remember about the array.
- $i$: the index (0 for the first item). Counting from 0 puts the first item exactly at $b$. Counting from 1 would need an extra subtraction.
- $s$: the size of one item in bytes. Items must all be the same size, otherwise the multiplication gives the wrong place. This is why a Python list stores equal-sized pointers, not the objects.
- One multiplication and one addition, however long the array is.

#flashcards/algorithms/example #flashcards/topic/data-structures $b=1000$, $s=4$, $i=3$. What is the address?::$1000+3\times4=1012$. The address comes from a calculation, not from walking past items 0, 1 and 2. A linked list has no such formula and needs 3 hops.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the bound on copying when a dynamic array grows and explain each symbol.
?
$$\text{copies}\ \le\ n\left(1+\frac1g+\frac1{g^2}+\cdots\right)=\frac{g}{g-1}\,n$$
- $n$: the number of appends. The total work is proportional to $n$, so each append costs a fixed amount on average: $O(1)$ amortised.
- $g$: the growth factor (2 in the picture, about 1.125 in Python's rule). A bigger $g$ means fewer copies but more unused room. With $g\to1$ (adding one slot at a time) the bound blows up and the total cost becomes $O(n^2)$.
- The brackets are a geometric series: the last copy moves at most $n$ items, the one before at most $n/g$, and so on.

#flashcards/algorithms/example #flashcards/topic/data-structures Doubling ($g=2$), $n=8$ appends from capacity 1. How many copies, and what is the bound? And for $n=1000$?::The array is full after 1, 2 and 4 items, so it copies $1+2+4=7$. The bound is $\frac21\times8=16$. For $n=1000$ the code counted 1,023 copies against a bound of 2,000. The total copying is less than twice the number of appends, which is what amortised $O(1)$ means.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the total cost of inserting $n$ items at the front one by one and explain each symbol.
?
$$\sum_{k=0}^{n-1}k=\frac{n(n-1)}{2}$$
- $k$: how many items are in the array just before the insert. Each has to shift one place, so one insert costs about $k$.
- The sum adds the cost of each insert, from $k=0$ (empty array) to $k=n-1$. The total grows like $n^2$.
- $\frac{n(n-1)}{2}$: the sum $0+1+\dots+(n-1)$ by the pairing trick (first plus last, and so on: each pair adds to $n-1$, and there are $n/2$ pairs). It is $O(n^2)$.

#flashcards/algorithms/example #flashcards/topic/data-structures $n=4$ and then $n=1000$. How many items are moved?::$n=4$: $0+1+2+3=6$, and $4\times3/2=6$. $n=1000$: the code counted 499,500 moves, against at most 2,000 copies for 1,000 appends, about 250 times more. This counts moves, not measured time.

#flashcards/algorithms/concept #flashcards/topic/data-structures Why is `append` cheap on a Python list?::The list keeps spare room after its last item, so `append` just fills the next free slot. When the spare room runs out it moves house: copies every item into a bigger block. That copy is slow but rare, so each append is cheap on average (amortised).

#flashcards/algorithms/concept #flashcards/topic/data-structures Why is adding at the front or middle of an array expensive?::Every item after the new one must shift along by one place to make room.

#flashcards/algorithms/reverse #flashcards/topic/data-structures Linked list:::Each item is a small box holding a value and the address of the next box. Adding at the front is one step, but reaching item $i$ means following $i$ links from the start.

#flashcards/algorithms/image #flashcards/topic/data-structures ![[dynamic-array.svg]]
What does this picture show?
?
A dynamic array growing as numbers are added. The black boxes hold data (the logical size). The dotted boxes are spare room. The whole row is the capacity. When the spare room runs out (the turtles mark the slow steps, at capacity 2 and then 4) the array moves to a block twice as big. (From the note's caption: Derrick Coetzee, Wikimedia Commons, CC0.)

#flashcards/algorithms/image #flashcards/topic/data-structures ![[singly-linked-list.svg]]
What does this picture show?
?
A singly linked list with three numbers (12, 99, 37). Each box holds a value and an arrow to the next box. The last box points to nothing (the crossed box). (From the note's caption: Lasindi, based on an image by Derrick Coetzee, Wikimedia Commons, public domain.)
