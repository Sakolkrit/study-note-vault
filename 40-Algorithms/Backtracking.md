---
tags: [algorithms, search, recursion, backtracking, constraint]
status: growing
level: 4
tier: 2
---

# Backtracking

**Backtracking** solves a puzzle by building the answer one choice at a time, and as soon as a choice cannot lead anywhere, it **undoes** it and tries the next one.

## 🟢 1. Simple (age 12)
Backtracking is how you get through a maze. At each fork you pick a path. If you hit a dead end, you walk back to the last fork and take a different path. You never try the rest of a path you already know is blocked.

**Example:** you are filling a Sudoku. You write a 4 in a box, then a few boxes later you find that no number fits. So you go back, rub out the 4 and try a 5.

## 🔵 2. How it works
1. Break the answer into a list of **choices**: which number goes in the first box, then the second, and so on.
2. Make the first choice.
3. **Check** that nothing is broken yet. If something is, stop here: no way of finishing this half-built answer can work.
4. If all is well, go on to the next choice, using the same steps again ([[Recursion]]).
5. If every choice has been made and nothing is broken, you have an answer.
6. If you run out of options at some step, **go back** one step, undo that choice and try the next option there.
7. The set of all half-built answers forms a tree ([[Trees]]), and backtracking walks that tree depth first, cutting off whole branches as soon as a check fails.
8. **The catch:** the tree can still be huge, so a hard puzzle can take a very long time. Checking early and often is what keeps it small.

![[eight-queens-solution.png]]
*One solution of the eight queens puzzle: eight queens on a chess board, none attacking another (no two share a row, a column or a diagonal). This is the very first solution that the backtracking code below finds, with the board's rows read from the top. Image: Encik Tekateki, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Solution_1_for_Eight_Queens.png), CC BY-SA 4.0. I checked the queen positions against the code's output; the caption's description of the puzzle is the standard one, not from the file text.*

## 🟠 3. Technical detail
- **The pattern:** choose, check, recurse, undo. The undo step ("backtrack") is what restores the board so the next option starts from a clean state.
- **Pruning:** the check runs on the half-built answer, so a clash is caught as early as possible and the whole branch below it is skipped.
- **Typical uses:** puzzles such as Sudoku and the eight queens, listing all subsets or orderings with a rule, and finding a path through a maze or [[Graphs|graph]].
- **Cost:** in the worst case it is still exponential (see [[Big-O]]), and the saving depends on how early the checks fail.
- **Related methods:** [[Dynamic Programming]] remembers sub-answers when the same part comes up again and again, which backtracking does not. A [[Greedy Algorithms|greedy]] method makes each choice once and never undoes it.
- **Where else it shows up:** the way a regular expression engine tries one way of matching and then another is also backtracking ([[Regular Expressions]]). The exponential blow-up there is the reason for the "catastrophic backtracking" bug; a [[Finite Automaton]] avoids it.

```python
def solve(n, stop_at_first=False):
    nodes, sols = 0, []
    def place(cols):
        nonlocal nodes
        nodes += 1
        r = len(cols)
        if r == n: sols.append(tuple(cols)); return stop_at_first
        for c in range(n):
            if all(c != cc and abs(c - cc) != r - rr for rr, cc in enumerate(cols)):   # keep only choices that do not clash yet
                cols.append(c)
                if place(cols): return True
                cols.pop()                                                                # undo the choice and try the next one
        return False
    place([])
    return sols, nodes
for n in (4, 6, 8):
    sols, nodes = solve(n)
    print(f"{n} queens: {len(sols)} solutions; backtracking looked at {nodes:,} partial boards in total")
sol, nodes = solve(8, True)
print("first solution found (column of the queen in each row):", sol[0], " after", f"{nodes:,}", "partial boards")
print("check:", all(sol[0][i] != sol[0][j] and abs(sol[0][i] - sol[0][j]) != j - i for i in range(8) for j in range(i + 1, 8)))
```

## 🔴 4. Deep / formal
**How big is the search?** For the $N$ queens puzzle, with one queen in each row, the number of boards to look at without any pruning is

$$
N^{N}\quad\text{(any column in each row)}\qquad\text{or}\qquad N!\quad\text{(no two in the same column)}
$$

**Where:**
- $N$ is the number of queens, which is also the board size.
- $N^N$ counts every way to choose one column for each of the $N$ rows. **Matters:** it grows faster than any fixed power of $N$, so brute force dies quickly.
- $N!$ (N factorial, $N\times(N-1)\times\dots\times1$) counts the boards in which no column repeats. **Matters:** it is a much smaller count, but it still grows faster than any fixed power of $N$.

**Example (tiny numbers):** for $N=4$, $N^N=4^4=\mathbf{256}$ and $N!=4\cdot3\cdot2\cdot1=\mathbf{24}$. For $N=8$: $8^8=\mathbf{16{,}777{,}216}$ and $8!=\mathbf{40{,}320}$. Meaning: forbidding repeated columns already cuts the work by a factor of 416 for $N=8$, and backtracking checks the diagonals as it goes, which cuts it a lot more (see the measurement below).

**Why a regular expression engine can blow up.** For the pattern `(a+)+b` on a text of $n$ letters `a` with no `b`, a backtracking engine may try every way of cutting the text into blocks before it fails. The number of ways is

$$
2^{\,n-1}
$$

**Where:**
- $n$ is the number of letters in the text. Between neighbouring letters there are $n-1$ gaps.
- Each gap is either "cut here" or "do not cut here". **Matters:** two choices per gap multiply, which is why the count doubles with every extra letter.

**Example (tiny numbers):** for $n=3$ (the text `aaa`) there are $2^2=\mathbf4$ ways: `a|a|a`, `a|aa`, `aa|a`, `aaa`. For $n=20$ there are $2^{19}=\mathbf{524{,}288}$ ways. Meaning: adding one letter doubles the work, so a text of 30 letters would need over 500 million tries. (This counts the ways to cut; it is the idea behind the slowdown, and I did not time a real engine.)

**Further notes:**
- **Choice of order:** trying the most constrained choice first usually prunes much earlier (from memory of common practice, not tested here).
- **Constraint propagation:** stronger solvers also rule out options in advance, not only after a clash.
- **Not the only answer:** for many puzzles a smarter method, such as dynamic programming, beats searching altogether.

**What came before, and how this improved on it:**
- **Before:** brute force: build every complete answer, then test each one.
- **Its limit:** the number of complete answers explodes. For 8 queens that is 16,777,216 boards, even before checking.
- **What changed:** test the half-built answer and abandon it at the first clash. The word "backtrack" is credited to D. H. Lehmer in the 1950s, and the method was described in print by Walker (1960) and by Golomb and Baumert (1965); all of these names and dates are from memory, not checked.
- **How much better:** in the code below, 8 queens by brute force means 16,777,216 boards, and with the no-shared-column rule 40,320. Backtracking looked at 2,057 partial boards and found all 92 solutions, which is 8,156 times fewer than brute force. **Honest limit:** the saving is large here but not magic, since the number of partial boards still grows quickly with $N$ (17 for 4 queens, 153 for 6, 2,057 for 8).

**Check it yourself:**

```python
import itertools
n = 8
brute = n ** n                                                         # every way to put one queen in each row, no checks
perms = sum(1 for _ in itertools.permutations(range(n)))               # also forbidding shared columns
def solve(n):
    nodes, sols = 0, 0
    def place(cols):
        nonlocal nodes, sols
        nodes += 1
        r = len(cols)
        if r == n: sols += 1; return
        for c in range(n):
            if all(c != cc and abs(c - cc) != r - rr for rr, cc in enumerate(cols)): place(cols + [c])
    place([]); return nodes, sols
nodes, sols = solve(n)
print(f"8 queens: all placements without checking {brute:,}; with no shared column {perms:,}; backtracking {nodes:,} partial boards")
print(f"all {sols} solutions found; backtracking looked at {brute // nodes:,}x fewer boards than brute force")
for m in (10, 20):
    print(f"splitting the text 'a' * {m} into blocks (what a backtracking regex tries before giving up on (a+)+b): {2 ** (m - 1):,} ways; a state machine reads it once: {m} steps")
```

## Related
- [[Recursion]] · [[Trees]] · [[Graphs]] · [[Dynamic Programming]] · [[Greedy Algorithms]] · [[Big-O]] · [[Stacks and Queues]] · [[Regular Expressions]] · [[Finite Automaton]]

## Sources
- The eight queens solution image: [Solution_1_for_Eight_Queens.png](https://commons.wikimedia.org/wiki/File:Solution_1_for_Eight_Queens.png) by Encik Tekateki, CC BY-SA 4.0.
- Lehmer, Walker, Golomb and Baumert as the sources of the word and the method: from memory, not opened.
- The code, the counts of partial boards and the 92 solutions were run for this note (plain Python). The regex slowdown is a count of ways to cut the text, not a timing of a real engine.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/algorithm-design Backtracking:::Getting through a maze. At each fork pick a path, and if you hit a dead end walk back to the last fork and take a different path. Never try the rest of a path you know is blocked.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design List the steps of backtracking.::Break the answer into choices, make the first choice, check nothing is broken (if it is, stop here), go on to the next choice (recursion), and when every choice is made with nothing broken you have an answer. If you run out of options, go back one step, undo that choice and try the next option.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design What tree does backtracking walk?::The set of all half-built answers forms a tree. Backtracking walks it depth first and cuts off whole branches as soon as a check fails.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design What is the catch of backtracking?::The tree can still be huge, so a hard puzzle can take a very long time. Checking early and often is what keeps it small.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the size of the search for $N$ queens by brute force and explain each symbol.
?
$$N^{N}\quad\text{(any column in each row)}\qquad\text{or}\qquad N!\quad\text{(no two in the same column)}$$
- $N$: the number of queens, which is also the board size.
- $N^N$: every way to choose one column for each of the $N$ rows. It grows faster than any fixed power of $N$, so brute force dies quickly.
- $N!$ ($N\times(N-1)\times\dots\times1$): the boards in which no column repeats. Much smaller, but still faster growth than any fixed power of $N$.

#flashcards/algorithms/example #flashcards/topic/algorithm-design Compare $N^N$ and $N!$ for $N=4$ and $N=8$.::$N=4$: $4^4=256$ and $4!=24$. $N=8$: $8^8=16{,}777{,}216$ and $8!=40{,}320$. Forbidding repeated columns already cuts the work by a factor of 416 for $N=8$, and checking diagonals as it goes cuts far more.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the number of ways to cut a text of $n$ letters into pieces and explain each symbol.
?
$$2^{\,n-1}$$
- $n$: the number of letters in the text. Between neighbouring letters there are $n-1$ gaps.
- Each gap is either "cut here" or "do not cut here". Two choices per gap multiply, so the count doubles with every extra letter.

#flashcards/algorithms/example #flashcards/topic/algorithm-design For `aaa` and for $n=20$, how many ways?::$n=3$: $2^2=4$ ways: `a|a|a`, `a|aa`, `aa|a`, `aaa`. $n=20$: $2^{19}=524{,}288$. A text of 30 letters would need over 500 million tries. This counts ways to cut and is the idea behind the slowdown. The note did not time a real engine.

#flashcards/algorithms/image #flashcards/topic/algorithm-design ![[eight-queens-solution.png]]
What does this picture show?
?
One solution of the eight queens puzzle: eight queens on a chess board, none attacking another (no two share a row, a column or a diagonal). It is the first solution the note's backtracking code finds, with the board's rows read from the top. (From the note's caption: Encik Tekateki, Wikimedia Commons.)

#flashcards/algorithms/cloze #flashcards/topic/algorithm-design Backtracking checks ==early and often==, so whole branches of the tree are cut off.
