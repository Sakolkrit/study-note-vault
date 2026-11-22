---
tags: [algorithms, recursion, divide-and-conquer, programming, python]
status: growing
level: 4
tier: 1
---

# Recursion

Recursion means **solving a problem by solving a smaller copy of the same problem**, and repeating until the problem is so small that the answer is obvious. A function that calls itself is recursive. It is a natural way to handle things that contain smaller versions of themselves: lists inside lists, folders inside folders, and problems you can split in two. The sorting methods in [[Sorting]] and the halving in [[Binary Search]] are both recursive ideas.

## 🟢 1. Simple (age 12)
Imagine a set of Russian nesting dolls, and you want to know how many dolls there are. You open the biggest one and find a smaller set inside. You do not need a new method: you ask the same question about the smaller set ("how many dolls are in here?") and add one for the doll you just opened. You keep going until you open a doll with nothing inside. That empty case is the end, and the answers then come back up the chain: 0, then 1, then 2, and so on.

**Example:** You are in a long queue and want to know your place. You ask the person in front, "what is your place?" They ask the person in front of them, and so on, up to the first person, who says "1". Then each person adds one and passes the answer back: "2", "3", ... until it reaches you.

## 🔵 2. How it works
Every recursive solution has two parts.
1. **A base case:** a small version of the problem that you can answer straight away without any more splitting (the empty doll, the first person in the queue). **Without it, the process never stops.**
2. **A recursive step:** the rule that makes the problem *smaller* and hands the smaller problem to the same method, then uses its answer to build the answer to the bigger one.

How it plays out:
1. You start with the full problem. It is too big to answer directly, so you make it a bit smaller and call the same method on that.
2. That call also makes the problem smaller and calls again. The calls pile up, one waiting for the next.
3. At last a call reaches the base case and answers directly.
4. The answers then come back up the pile. Each waiting call uses the answer it received to finish its own job.
5. The first call gets its answer and you are done.

The computer keeps the waiting calls on a pile called the **call stack**, like a pile of plates: the last plate added is the first removed.

![[tower-of-hanoi-3-discs.gif]]
*The Tower of Hanoi puzzle with 3 discs, a classic recursive problem. The aim is to move all the discs from the left peg to the right peg, one at a time, never putting a bigger disc on a smaller one. The recursive idea: to move a tower of 3, first move the top 2 out of the way, then move the biggest disc, then move the 2 back on top. Each of those "move 2" jobs is the same puzzle, only smaller. It takes 7 moves (an animation: some viewers show only the first frame). Image: André Karwath (Aka), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Tower_of_Hanoi.gif), CC BY-SA 2.5.*

## 🟠 3. Technical detail
- **The two parts again:** every correct recursive function has (1) a base case and (2) a step that moves *toward* the base case. A missing base case, or a step that does not shrink the problem, gives endless recursion. Python stops this with a `RecursionError` (the code shows it).
- **The call stack has a limit:** each pending call uses memory. Python allows a depth of 1,000 calls by default (the code reads the limit and gets 1000, and fails at a depth of 100,000). Very deep jobs should use a loop or an explicit stack of your own (the code shows the same walk with both).
- **Recursion against loops:** anything recursive can be written as a loop (with your own stack if needed), and the other way round. Pick the one that reads more clearly. Recursion is clearest for tree-like data and for splitting problems. A loop is simpler and cheaper for plain counting.
- **How much work is it?** For recursion, you write the cost as a *recurrence*: the cost of a problem of size $n$ in terms of the cost of smaller ones. Solving it gives the Big-O (see [[Big-O]]). Three common ones, all checked in the code:
  - $T(n)=T(n/2)+1$ is $O(\log n)$ (binary search).
  - $T(n)=2T(n/2)+n$ is $O(n\log n)$ (merge sort, equation 2 below).
  - $T(n)=2T(n-1)+1$ is $O(2^n)$ (Tower of Hanoi, equation 1 below).
- **Repeated work:** if the smaller problems overlap, a plain recursion solves the same subproblem again and again. The naive Fibonacci recursion is the standard example: 242,785 calls for $n=25$, though only 26 different values are needed. The fix is to **remember answers already worked out** (*memoisation*), which cuts the calls to about $n$ (equation 3 below). Python's `functools.lru_cache` does this in one line. Building the table of answers from the bottom up on purpose is *dynamic programming* (see [[Dynamic Programming]]).
- **Divide and conquer:** split a problem into independent parts, solve each (by recursion), and combine the results. Merge sort, quicksort and binary search are of this kind (see [[Sorting]] and [[Binary Search]]).
- **Backtracking and trees:** to try all possibilities (all orderings of 4 items, all subsets), recursion explores a *tree* of choices. The code counts $4!=24$ orderings and $2^4=16$ subsets. These grow very fast, so they are only practical for small inputs.
- **In this vault:** a [[Decision Tree]] is a recursive structure: each branch is a smaller decision tree. [[Backpropagation]] walks a computation graph backwards, which can be written recursively (see [[PyTorch]]). The Bellman equation in [[Q-Learning]] defines the value of a state in terms of the values of the next states, a recursion over time.
- **Tail calls:** some languages reuse the same stack space when the recursive call is the very last thing a function does. Python does *not* do this (from memory; consistent with the failure of `count_down(100000)` in the code, but I did not test a tail-call version separately).
- **Limits:** the code runs small examples. It does not time anything, and it does not test other languages.

The code covers six things: (1) factorial and the recursion limit, (2) the Tower of Hanoi, (3) Fibonacci with and without remembering answers, (4) nested data and choices, (5) the three recurrences, (6) a walk through a tree with a recursive call and with an explicit stack.

```python
import sys, functools
# 1) the smallest recursion: a base case that stops it, and a step that moves toward it
def fact(n): return 1 if n <= 1 else n * fact(n - 1)
def fact_loop(n):
    r = 1
    for i in range(2, n + 1): r *= i
    return r
print('fact(5) =', fact(5), '| loop gives', fact_loop(5), '| fact(20) equals loop:', fact(20) == fact_loop(20))
def count_down(n): return 0 if n == 0 else 1 + count_down(n - 1)
print('Python stops runaway recursion at a limit of', sys.getrecursionlimit(), 'calls deep; count_down(500) =', count_down(500))
try: count_down(10 ** 5)
except RecursionError: print('count_down(100000) -> RecursionError (too deep, so use a loop or a stack for very deep jobs)')
def no_base(n): return no_base(n - 1)                        # a missing base case never stops
try: no_base(5)
except RecursionError: print('no base case -> RecursionError')
# 2) Tower of Hanoi: move n discs from peg A to C using B; the recursion mirrors the idea exactly
def hanoi(n, a, b, c, moves):
    if n == 0: return
    hanoi(n - 1, a, c, b, moves); moves.append((a, c)); hanoi(n - 1, b, a, c, moves)     # clear the way, move the big disc, rebuild on top
m = []; hanoi(3, 'A', 'B', 'C', m); print('3 discs:', len(m), 'moves:', ' '.join(a + c for a, c in m))
for n in (1, 2, 3, 10, 20):
    m = []; hanoi(n, 'A', 'B', 'C', m); print('discs', n, '| moves', len(m), '| 2^n - 1 =', 2 ** n - 1)
print('64 discs at one move per second would take about', round((2 ** 64 - 1) / (365.25 * 24 * 3600) / 1e9), 'billion years')
# 3) the same recursion can repeat work: Fibonacci with and without remembering answers
calls = {'n': 0}
def fib(n):
    calls['n'] += 1
    return n if n < 2 else fib(n - 1) + fib(n - 2)
def Fib(n):                                                  # the Fibonacci number itself, by a loop, to check the call counts
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a
for n in (5, 10, 25):
    calls['n'] = 0; fib(n); print('naive fib', n, '| calls', calls['n'], '| 2*F(n+1)-1 =', 2 * Fib(n + 1) - 1)
runs = {'n': 0}
@functools.lru_cache(maxsize=None)
def fib_memo(n):
    runs['n'] += 1
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)
for n in (5, 25, 80):
    runs['n'] = 0; fib_memo.cache_clear(); v = fib_memo(n); print('memoised fib', n, '=', v, '| function bodies run', runs['n'], '(n + 1 =', n + 1, ') | matches loop:', v == Fib(n))
# 4) recursion for nested data and for choices
def flatten(x): return [y for item in x for y in flatten(item)] if isinstance(x, list) else [x]
nested = [1, [2, [3, [4]], 5], [[6]]]; print('flatten', nested, '->', flatten(nested))
def subsets(items): return [[]] if not items else (lambda rest: rest + [[items[0]] + s for s in rest])(subsets(items[1:]))
def perms(items): return [items] if len(items) <= 1 else [[x] + p for i, x in enumerate(items) for p in perms(items[:i] + items[i + 1:])]
print('subsets of 4 items:', len(subsets([1, 2, 3, 4])), '(2^4) | orderings of 4 items:', len(perms([1, 2, 3, 4])), '(4!) | orderings of 3:', perms([1, 2, 3]))
# 5) the "how much work" recurrences, counted
def T_halve_and_scan(n): return 0 if n == 1 else 2 * T_halve_and_scan(n // 2) + n          # merge-sort style: two halves plus n work
def T_halve(n): return 0 if n == 1 else T_halve(n // 2) + 1                                  # binary-search style
print('T(n)=2T(n/2)+n at n=8, 1024:', T_halve_and_scan(8), T_halve_and_scan(1024), '(n*log2 n = 24, 10240) | T(n)=T(n/2)+1 at 1024:', T_halve(1024), '(log2 n = 10)')
# 6) the call stack made visible: the same job with an explicit stack instead of recursive calls
def depth_first(tree):
    order, stack = [], [tree]
    while stack:
        node = stack.pop(); order.append(node[0]); stack.extend(reversed(node[1]))
    return order
def depth_first_rec(node): return [node[0]] + [x for ch in node[1] for x in depth_first_rec(ch)]
tr = ('a', [('b', [('d', []), ('e', [])]), ('c', [('f', [])])]); print('depth-first, stack:', depth_first(tr), '| recursive:', depth_first_rec(tr), '| same:', depth_first(tr) == depth_first_rec(tr))
# fact(5) = 120 | loop gives 120 | fact(20) equals loop: True
# Python stops runaway recursion at a limit of 1000 calls deep; count_down(500) = 500
# count_down(100000) -> RecursionError (too deep, so use a loop or a stack for very deep jobs)
# no base case -> RecursionError
# 3 discs: 7 moves: AC AB CB AC BA BC AC
# discs 1 | moves 1 | 2^n - 1 = 1
# discs 2 | moves 3 | 2^n - 1 = 3
# discs 3 | moves 7 | 2^n - 1 = 7
# discs 10 | moves 1023 | 2^n - 1 = 1023
# discs 20 | moves 1048575 | 2^n - 1 = 1048575
# 64 discs at one move per second would take about 585 billion years
# naive fib 5 | calls 15 | 2*F(n+1)-1 = 15
# naive fib 10 | calls 177 | 2*F(n+1)-1 = 177
# naive fib 25 | calls 242785 | 2*F(n+1)-1 = 242785
# memoised fib 5 = 5 | function bodies run 6 (n + 1 = 6 ) | matches loop: True
# memoised fib 25 = 75025 | function bodies run 26 (n + 1 = 26 ) | matches loop: True
# memoised fib 80 = 23416728348467685 | function bodies run 81 (n + 1 = 81 ) | matches loop: True
# flatten [1, [2, [3, [4]], 5], [[6]]] -> [1, 2, 3, 4, 5, 6]
# subsets of 4 items: 16 (2^4) | orderings of 4 items: 24 (4!) | orderings of 3: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
# T(n)=2T(n/2)+n at n=8, 1024: 24 10240 (n*log2 n = 24, 10240) | T(n)=T(n/2)+1 at 1024: 10 (log2 n = 10)
# depth-first, stack: ['a', 'b', 'd', 'e', 'c', 'f'] | recursive: ['a', 'b', 'd', 'e', 'c', 'f'] | same: True
```
Reading the output (Python 3.9, standard library only):
- **Base case and limit:** `fact(5)` is 120, and the recursive and loop versions agree even for 20. Python's default depth limit is 1,000. A countdown from 500 works, from 100,000 raises `RecursionError`, and a function with no base case also raises it.
- **Tower of Hanoi:** the recursive solution for 3 discs makes 7 moves, `AC AB CB AC BA BC AC` (disc 1 from A to C, disc 2 from A to B, disc 1 from C to B, and so on). In the animation the smallest disc also goes to the target peg first, then the middle disc to the middle peg. The move counts are $2^n-1$ for every size tried: 1, 3, 7, 1,023 and 1,048,575 for 1, 2, 3, 10 and 20 discs.
- **A big number:** with 64 discs, $2^{64}-1$ moves at one move per second would take about 585 billion years. That is plain arithmetic (I assumed one move per second and a year of 365.25 days).
- **Repeated work:** naive Fibonacci made 15, 177 and 242,785 calls for $n=5$, 10 and 25, which match the formula $2F(n+1)-1$ exactly. With remembered answers, the function body ran only $n+1$ times (6, 26 and 81), and the answers match the loop, even for $n=80$, where the naive way would need about $7.6\times10^{16}$ calls (from the formula $2F(81)-1$, not run).
- **Nested data:** the flatten function turns `[1, [2, [3, [4]], 5], [[6]]]` into `[1, 2, 3, 4, 5, 6]`. For 4 items there are 16 subsets ($2^4$) and 24 orderings ($4!$).
- **Recurrences:** $T(n)=2T(n/2)+n$ gives 24 at $n=8$ and 10,240 at $n=1024$, which equal $n\log_2n$. $T(n)=T(n/2)+1$ gives 10 at 1,024, which is $\log_2n$.
- **Stack against recursion:** the explicit-stack walk and the recursive walk visit the tree in the same order, a, b, d, e, c, f.
- **Limits:** only small examples. I did not time anything and did not test other languages or tail-call behaviour.

## 🔴 4. Deep / formal

**Hanoi's move count.** To move $n$ discs from one peg to another, the recursion moves the top $n-1$ discs out of the way, moves the biggest disc, and moves the $n-1$ discs back on top. The number of moves $M(n)$ therefore satisfies

$$M(0)=0,\qquad M(n)=2\,M(n-1)+1\quad\Longrightarrow\quad M(n)=2^{n}-1$$

**Where:**
- $M(0)=0$ is the base case: with no discs, nothing needs moving. **Matters:** it stops the recursion and anchors the formula.
- $M(n-1)$ is the cost of moving the smaller tower, and it appears twice. **Matters:** two copies of a problem only one disc smaller is what makes the cost double with every disc.
- The $+1$ is the single move of the biggest disc. **Matters:** it is the only move not done by a smaller copy.
- $2^n-1$ is the closed form. **Matters:** the cost is exponential, so each extra disc doubles the work, and 64 discs are out of reach.
- To see the closed form: adding 1 to both sides of the recurrence gives $M(n)+1=2\,(M(n-1)+1)$, so $M(n)+1$ doubles each step from $M(0)+1=1$, which gives $2^n$. I checked the formula by running the recursion; the algebra is a short standard argument that I did not prove further.

**Example (tiny numbers):**
1. $M(1)=2\cdot0+1=1$. $M(2)=2\cdot1+1=3$. $M(3)=2\cdot3+1=7$.
2. The code's recursion made 7 moves for 3 discs, and 1,023 for 10, and 1,048,575 for 20. These all equal $2^n-1$.
3. Meaning: each extra disc doubles the number of moves. 20 discs need about a million moves, and 64 discs need $2^{64}-1\approx1.8\times10^{19}$, which would take about 585 billion years at one move per second. Some problems are simple to describe by recursion but too costly to solve for large inputs.

**Divide and conquer.** If a problem of size $n$ is split into two halves, each solved the same way, and combining the answers costs $n$ steps, the total cost satisfies

$$T(1)=0,\qquad T(n)=2\,T(n/2)+n\quad\Longrightarrow\quad T(n)=n\log_2n\ \ (n\text{ a power of }2)$$

**Where:**
- $T(1)=0$ is the base case: a single item needs no work. **Matters:** it ends the splitting.
- $2\,T(n/2)$ is the cost of the two half-size sub-problems. **Matters:** the sizes halve at each level, so there are $\log_2n$ levels.
- $+n$ is the cost of combining the halves (for merge sort, merging). **Matters:** each level of the splitting has total combining cost $n$ (two pieces of $n/2$, four of $n/4$, ...). With $\log_2n$ levels, that makes $n\log_2n$ in total.
- The formula holds exactly when $n$ is a power of 2, and approximately otherwise. **Matters:** the code checks powers of 2 only (8 and 1,024).
- The same recurrence form with a different combining cost gives other results: a combining cost of 1 and one half gives $T(n)=T(n/2)+1=\log_2n$ (binary search).

**Example (tiny numbers):** $n=8$.
1. $T(2)=2\,T(1)+2=2$. $T(4)=2\,T(2)+4=8$. $T(8)=2\,T(4)+8=24$.
2. Compare $n\log_2n=8\cdot3=24$. They match. For $n=1024$ the code got 10,240, which equals $1024\cdot10$.
3. Meaning: the cost of "split in half, solve each half, combine" is $n$ per level over $\log_2n$ levels. This is why merge sort is fast and why 1,024 items need about ten thousand steps, not a million ($1024^2$).

**The cost of repeated work.** Let $C(n)$ be the number of calls the naive recursive Fibonacci makes for input $n$. Each call makes two more calls and counts itself:

$$C(0)=C(1)=1,\qquad C(n)=C(n-1)+C(n-2)+1\quad\Longrightarrow\quad C(n)=2F(n+1)-1,\qquad \text{memoised: } n+1\ \text{bodies run}$$

**Where:**
- $F(n)$ is the $n$-th Fibonacci number ($F(0)=0$, $F(1)=1$, $F(2)=1$, $F(3)=2,\dots$). **Matters:** the number of calls is itself a Fibonacci number, so it grows by a factor of about 1.618 for each step of $n$ (exponential growth).
- $C(n-1)+C(n-2)$ is the work of the two sub-calls, and the $+1$ is the call itself. **Matters:** the two sub-calls overlap heavily (both need $F(n-2)$), and nothing is shared.
- The memoised version runs the body once for each $n$ from 0 to $n$. **Matters:** the second time a value is needed, it is read from memory in one step, so the total is linear: $n+1$ bodies (and $2n-1$ calls in all, counting the lookups; I did not count those).
- The closed form $2F(n+1)-1$ was checked on $n=5$, 10 and 25. **Matters:** it explains the 242,785 calls at $n=25$ against the 26 bodies needed with memoisation.

**Example (tiny numbers):** $n=5$.
1. $C(2)=1+1+1=3$, $C(3)=3+1+1=5$, $C(4)=5+3+1=9$, $C(5)=9+5+1=15$.
2. Formula: $F(6)=8$, so $2\cdot8-1=15$. The code counted 15 calls.
3. Memoised: the body runs for $n=0,1,2,3,4,5$, which is 6 times. The code counted 6.
4. Meaning: at $n=5$ the saving is small (15 against 6). At $n=25$ it is 242,785 against 26, and at $n=50$ it would be about $4\times10^{10}$ against 51 (from the formula; not run). Remembering answers turns an exponential method into a linear one when sub-problems overlap.

**Other points (from memory, not checked here):**
- The Tower of Hanoi puzzle is usually credited to the French mathematician Édouard Lucas (1883), with a legend about monks moving 64 discs. I did not check the history or the legend.
- Python's recursion limit is a setting, not a law: it can be raised (`sys.setrecursionlimit`), but a very deep recursion can crash the interpreter, so a loop or explicit stack is safer. I did not test raising it.
- Recursion and induction are closely linked: a recursive method is correct if the base case is right and the step is right *assuming the smaller case is*. I have not covered the proofs here.
- The idea of a function defined in terms of itself is central to many areas (grammars, trees, functional programming). I covered only the programming side.

**What came before, and how this improved on it:**
- **Before:** loops written for a fixed shape, one loop per level.
- **Its limit:** a loop version does not fit data of unknown depth (a list inside a list inside a list), because you cannot know how many loops to write.
- **What changed:** let a function call itself on the smaller part, with a base case that stops it. Recursive definitions of functions go back to the early 1900s, and the idea entered programming languages with LISP in 1958 (from memory, not checked).
- **How much better:** in the code below, a recursive `flatten` of 5 levels of nested lists gave `[1, 2, 3, 4, 5]`, while a two-loop version stopped after two levels. The towers of Hanoi puzzle with 20 discs needs 1,048,575 moves, and the 3-line recursive rule produces exactly the count $2^n-1$. The honest cost: Python stops deep recursion near 1,000 levels by default.

**Check it yourself:**

```python
def flatten(x):
    return [y for item in x for y in flatten(item)] if isinstance(x, list) else [x]
def two_loops(x):                     # fixed depth: only handles lists of lists
    return [y for a in x for y in (a if isinstance(a, list) else [a])]
deep = [1, [2, [3, [4, [5]]]]]
print("nested list:", deep)
print("recursive flatten:", flatten(deep))
print("two-loop version: ", two_loops(deep), "(stops after two levels)")
moves = []
def hanoi(n, a, b, c):
    if n == 0: return
    hanoi(n - 1, a, c, b); moves.append((a, c)); hanoi(n - 1, b, a, c)
for n in (3, 10, 20):
    moves.clear(); hanoi(n, "A", "B", "C"); print(f"towers with {n} discs: {len(moves):,} moves, rule 2^n - 1 = {2**n - 1:,}")
```

## Related
[[Big-O]] · [[Sorting]] · [[Binary Search]] · [[Dynamic Programming]] · [[Hash Tables]] · [[Python]] · [[Decision Tree]] · [[Backpropagation]] · [[PyTorch]] · [[Q-Learning]] · [[Stacks and Queues]] · [[Graphs]] · [[Trees]] · [[Regular Expressions]]
- [[Backtracking]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, on recurrences and divide and conquer (from memory; I did not open it for this note).
- Python documentation for `sys.getrecursionlimit` and `functools.lru_cache` (docs.python.org).
- Figure: [Tower_of_Hanoi.gif](https://commons.wikimedia.org/wiki/File:Tower_of_Hanoi.gif) (André Karwath, CC BY-SA 2.5), on Wikimedia Commons. I viewed 12 of its 20 frames before writing the caption.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/algorithm-design Recursion:::To count the dolls in a nest, open the biggest, ask the same question about the smaller set inside, and add one. You keep going until a doll has nothing inside.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design What two parts does every recursive solution have?::A base case (a small version you can answer straight away, without it the process never stops) and a recursive step (make the problem smaller, hand it to the same method, and use its answer to build the bigger one).

#flashcards/algorithms/concept #flashcards/topic/algorithm-design What is the call stack?::The pile where the computer keeps the waiting calls, like plates: the last plate added is the first removed. Calls pile up until one reaches the base case, then answers come back up the pile.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the Tower of Hanoi move count and explain each symbol.
?
$$M(0)=0,\qquad M(n)=2\,M(n-1)+1\quad\Longrightarrow\quad M(n)=2^{n}-1$$
- $M(0)=0$: the base case, nothing to move with no discs. It stops the recursion.
- $M(n-1)$: the cost of moving the smaller tower, which appears twice. Two copies of a problem only one disc smaller make the cost double with every disc.
- The $+1$: the single move of the biggest disc, the only move not done by a smaller copy.
- $2^n-1$: the closed form. The cost is exponential, so each extra disc doubles the work.

#flashcards/algorithms/example #flashcards/topic/algorithm-design What are $M(1)$, $M(2)$, $M(3)$, and how long would 64 discs take?::$M(1)=1$, $M(2)=3$, $M(3)=7$. The code made 7, 1,023 and 1,048,575 moves for 3, 10 and 20 discs. 64 discs need $2^{64}-1\approx1.8\times10^{19}$ moves, about 585 billion years at one move per second.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the cost of split-in-half-and-combine and explain each symbol.
?
$$T(1)=0,\qquad T(n)=2\,T(n/2)+n\quad\Longrightarrow\quad T(n)=n\log_2n\ \ (n\text{ a power of }2)$$
- $T(1)=0$: the base case, a single item needs no work.
- $2\,T(n/2)$: the cost of the two half-size sub-problems. The sizes halve at each level, so there are $\log_2n$ levels.
- $+n$: the cost of combining the halves (merging, for merge sort). Each level has total combining cost $n$, so the total is $n\log_2n$.
- The formula is exact when $n$ is a power of 2, approximate otherwise.

#flashcards/algorithms/example #flashcards/topic/algorithm-design $n=8$ and $n=1024$. What is $T$?::$T(2)=2$, $T(4)=8$, $T(8)=24=8\cdot3$. For $n=1024$ the code got 10,240, which is $1024\cdot10$. About ten thousand steps, not a million ($1024^2$). This is why merge sort is fast.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the number of calls of naive Fibonacci and explain each symbol.
?
$$C(0)=C(1)=1,\qquad C(n)=C(n-1)+C(n-2)+1\quad\Longrightarrow\quad C(n)=2F(n+1)-1,\qquad \text{memoised: } n+1\ \text{bodies run}$$
- $F(n)$: the $n$-th Fibonacci number ($F(0)=0$, $F(1)=1$, $F(2)=1$, $F(3)=2,\dots$). The number of calls is itself a Fibonacci number, growing by a factor of about 1.618 per step of $n$.
- $C(n-1)+C(n-2)$: the work of the two sub-calls, and the $+1$ is the call itself. The sub-calls overlap heavily (both need $F(n-2)$) and nothing is shared.
- The memoised version runs the body once for each $n$ from 0 to $n$. The second time a value is needed it is read from memory.

#flashcards/algorithms/example #flashcards/topic/algorithm-design $n=5$. How many calls naive and memoised? And $n=25$?::$C(2)=3$, $C(3)=5$, $C(4)=9$, $C(5)=15=2F(6)-1=2\cdot8-1$ (the code counted 15). Memoised: the body runs 6 times (the code counted 6). At $n=25$: 242,785 against 26. At $n=50$ about $4\times10^{10}$ against 51 (from the formula, not run). Remembering answers turns an exponential method into a linear one when sub-problems overlap.

#flashcards/algorithms/image #flashcards/topic/algorithm-design ![[tower-of-hanoi-3-discs.gif]]
What does this picture show?
?
The Tower of Hanoi with 3 discs. The aim is to move all discs from the left peg to the right peg, one at a time, never putting a bigger disc on a smaller one. The recursive idea: to move a tower of 3, first move the top 2 out of the way, then the biggest disc, then the 2 back on top. Each "move 2" job is the same puzzle, only smaller. It takes 7 moves. It is an animation, and some viewers show only the first frame. (From the note's caption: André Karwath, Wikimedia Commons.)

#flashcards/algorithms/cloze #flashcards/topic/algorithm-design Without a ==base case==, a recursion never stops.
