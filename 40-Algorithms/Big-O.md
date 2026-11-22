---
tags: [algorithms, complexity, big-o, programming, computer-science]
status: growing
level: 4
tier: 1
---

# Big-O

Big-O is a way to describe **how the work an algorithm does grows as the input gets bigger**. It ignores small details like how fast your computer is, and keeps only the shape of the growth: does the work double when the input doubles, or does it quadruple, or does it hardly change? It is the standard language for comparing algorithms, and the reason why people say things like "attention costs n²" (see [[Self-Attention]]).

## 🟢 1. Simple (age 12)
Suppose you are looking for your name on a list of students. If the list has 10 names, you check at most 10. If it has 1,000 names, you may have to check 1,000. The work grows in step with the length of the list. Now suppose the list is sorted alphabetically and you open it in the middle, then in the middle of the half that is left, and so on. Even with 1,000 names, you would only need about 10 looks. Big-O is a short way of saying which kind of search you are doing: **one that grows with the list, or one that hardly grows at all**.

**Example:** Everyone in a class of 30 shakes hands with every other person. That is a lot of handshakes (435). If the class doubles to 60, the handshakes do not double. They go up about four times (1,770). A small change in the input made a big change in the work. Big-O gives that behaviour a name: "grows like n squared".

## 🔵 2. How it works
1. **Pick the size of the input,** and call it $n$. It might be the number of names, the number of numbers in a list, or the number of words in a text.
2. **Count the basic steps** the algorithm takes for an input of that size. A step is one small action, such as one comparison or one addition.
3. **Ask how the count grows** when $n$ grows. Try $n=10$, $n=100$, $n=1000$ and watch.
4. **Keep only the biggest part.** If the count is $3n^2+10n+5$, then for large $n$ the $n^2$ part dominates the others. Drop the smaller parts and the constant multiplier, and say "order $n^2$", written $O(n^2)$.
5. **Name the shape.** The common ones are: $O(1)$ (constant: does not grow), $O(\log n)$ (grows very slowly), $O(n)$ (grows in step with $n$), $O(n\log n)$ (a bit more than $n$), $O(n^2)$ (grows with the square), and $O(2^n)$ (explodes).
6. **Compare algorithms** by their shapes. For large inputs, a slower-growing shape will win in the end, even if it looks slower on tiny inputs.

![[big-o-growth-comparison.svg]]
*How the number of steps $N$ grows as the input size $n$ grows from 0 to 100, for the common shapes, with every constant set to 1. The flat line at the bottom is $1$. Slightly above it are $\log_2 n$ and $\sqrt n$, then the straight line $n$, then $n\log_2 n$ climbing steeply. The curves $n^2$, $2^n$ and $n!$ shoot off the top of the chart almost at once. Image: Cmglee, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Comparison_computational_complexity.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **What "step" means:** a basic operation that takes about the same time whatever the data (an addition, a comparison, reading an item by its position in a list). Big-O counts these and ignores how long each takes on a particular machine.
- **Usually the worst case:** unless said otherwise, Big-O describes the *most work* for an input of size $n$ (for example, looking for a name that is last, or not there). Some algorithms are also described by their *average* or *best* case, but these must be named.
- **Time and space:** the same notation describes memory as well as time. An algorithm can be fast but use a lot of memory (see [[GPU]] for how memory limits machine learning).
- **Common shapes and what makes them:**
  - $O(1)$: reading item number 7 of a list, or looking up a key in a hash table (usually; see [[Hash Tables]]).
  - $O(\log n)$: **halving** the problem each time, as in binary search (see [[Binary Search]]). A million items need about 20 steps.
  - $O(n)$: one pass over the data, such as summing a list or finding the largest.
  - $O(n\log n)$: the best general sorting methods (see [[Sorting]]).
  - $O(n^2)$: **a loop inside a loop** over the same data, such as comparing every pair. Attention is of this kind (see [[Self-Attention]]).
  - $O(2^n)$: trying every subset, or a recursion that branches twice without sharing work, such as the naive Fibonacci in the code (see [[Recursion]]).
- **Doubling test:** double $n$ and see what happens to the work. It doubles: $O(n)$. It quadruples: $O(n^2)$. It adds one step: $O(\log n)$. It squares itself: exponential. The code uses this.
- **Constants still matter in practice:** $O(n)$ with a large constant can be slower than $O(n\log n)$ with a small one, for realistic sizes. Big-O is about growth, not exact time. Measuring real programs on real data is a separate task.
- **Amortised cost:** an operation that is usually cheap but occasionally expensive can be $O(1)$ *on average over many calls*. A list that doubles its storage when it fills up is the classic case: each append is cheap, and the rare copy is spread over all the appends (the code counts it).
- **Picking the right data structure** often changes the Big-O. Checking "is this item in my collection?" takes about $n$ steps in a plain list and about 1 step in a set (the code counts a related case). This is why [[Pandas]] and [[NumPy]] are fast: their loops run in compiled code and they use structures with good complexity.
- **Where this shows up in this vault:** attention is $O(n^2)$ in the text length (see [[Self-Attention]]), and tricks such as [[FlashAttention]] reduce the *memory* but not the arithmetic. [[k-NN]] search over $n$ items is $O(n)$ per query unless an index is used ([[Vector Database]]). Matrix multiplication of $n\times n$ grids is about $O(n^3)$ with the simple method (see [[Matrix]]).
- **Not covered:** the formal classes of hard problems (such as "NP-complete"), lower bounds, and how to analyse randomised algorithms. These are bigger topics.

The code counts steps instead of timing them, because step counts are the same on every computer. It checks six things: (1) the steps of linear search, binary search and "compare all pairs"; (2) the doubling test; (3) the formal definition on a worked example; (4) two ways to find two numbers that add to a target; (5) an exponential recursion; and (6) the copying cost of a growing list.

```python
import math, random
# 1) count the steps instead of timing them: linear search, binary search, all pairs
def linear_steps(a, target):
    steps = 0
    for v in a:
        steps += 1
        if v == target: break
    return steps
def binary_steps(a, target):
    lo, hi, steps = 0, len(a) - 1, 0
    while lo <= hi:
        steps += 1; mid = (lo + hi) // 2
        if a[mid] == target: return steps
        if a[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return steps
def pair_steps(a):
    steps = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)): steps += 1
    return steps
for n in (10, 100, 1000, 10000):
    a = list(range(n)); worst = max(binary_steps(a, v) for v in range(-1, n + 1))   # try every target, present or not: the longest path
    print(n, '| linear worst', linear_steps(a, -1), '| binary worst', worst, '(formula', int(math.log2(n)) + 1, ') | all pairs', pair_steps(a) if n <= 1000 else n * (n - 1) // 2)
# 2) doubling the input: how the step count grows
print('ratio when n doubles (1000 -> 2000): linear', linear_steps(list(range(2000)), -1) / linear_steps(list(range(1000)), -1), '| pairs', round(pair_steps(list(range(2000))) / pair_steps(list(range(1000))), 2), '| binary adds', max(binary_steps(list(range(2000)), v) for v in range(-1, 2001)) - max(binary_steps(list(range(1000)), v) for v in range(-1, 1001)))
# 3) the definition: f(n) = 3n^2 + 10n + 5 is O(n^2): find c and n0 such that f(n) <= c*n^2 for all n >= n0
f = lambda n: 3 * n * n + 10 * n + 5
c, n0 = 4, 11
print('c=4, n0=11 works up to 100000:', all(f(n) <= c * n * n for n in range(n0, 100001)), '| fails at n=10:', f(10) > c * 100, '| c=3 never works at n=1000:', f(1000) > 3 * 1000 ** 2)
print('f(n)/n grows (so not O(n)):', [round(f(n) / n) for n in (10, 100, 1000, 10000)], '| f(n)/n^2 settles near 3:', [round(f(n) / n ** 2, 4) for n in (10, 100, 1000, 10000)])
# 4) two ways to find a pair that sums to a target: all pairs against a set of numbers seen so far
def two_sum_pairs(a, t):
    steps = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            steps += 1
            if a[i] + a[j] == t: return steps
    return steps
def two_sum_set(a, t):
    seen, steps = set(), 0
    for v in a:
        steps += 1
        if t - v in seen: return steps
        seen.add(v)
    return steps
random.seed(0); a = random.sample(range(10 ** 6, 2 * 10 ** 6), 2000)                      # all values >= 1e6, so target 5 never occurs
print('no pair found, n = 2000: all-pairs steps', two_sum_pairs(a, 5), '| set steps', two_sum_set(a, 5))
# 5) exponential: naive recursive Fibonacci counts its own calls
calls = {'n': 0}
def fib(n):
    calls['n'] += 1
    return n if n < 2 else fib(n - 1) + fib(n - 2)
for n in (10, 20, 25): calls['n'] = 0; v = fib(n); print('fib', n, '=', v, 'took', calls['n'], 'calls')
# 6) doubling the storage when it fills up: the copying cost of n appends (amortised O(1) each)
def copies(n):
    cap, size, total = 1, 0, 0
    for _ in range(n):
        if size == cap: total += size; cap *= 2
        size += 1
    return total
print('copies for 1000 appends:', copies(1000), '| per append:', round(copies(1000) / 1000, 3), '| for 1,000,000:', round(copies(10 ** 6) / 10 ** 6, 3))
# 10 | linear worst 10 | binary worst 4 (formula 4 ) | all pairs 45
# 100 | linear worst 100 | binary worst 7 (formula 7 ) | all pairs 4950
# 1000 | linear worst 1000 | binary worst 10 (formula 10 ) | all pairs 499500
# 10000 | linear worst 10000 | binary worst 14 (formula 14 ) | all pairs 49995000
# ratio when n doubles (1000 -> 2000): linear 2.0 | pairs 4.0 | binary adds 1
# c=4, n0=11 works up to 100000: True | fails at n=10: True | c=3 never works at n=1000: True
# f(n)/n grows (so not O(n)): [40, 310, 3010, 30010] | f(n)/n^2 settles near 3: [4.05, 3.1005, 3.01, 3.001]
# no pair found, n = 2000: all-pairs steps 1999000 | set steps 2000
# fib 10 = 55 took 177 calls
# fib 20 = 6765 took 21891 calls
# fib 25 = 75025 took 242785 calls
# copies for 1000 appends: 1023 | per append: 1.023 | for 1,000,000: 1.049
```
Reading the output (Python 3.9, no libraries, and no randomness except one fixed-seed list):
- **Three shapes side by side:** for a list of 1,000 numbers, linear search needs up to 1,000 steps, binary search at most 10, and comparing all pairs needs 499,500. At 10,000 numbers: 10,000, 14 and 49,995,000. The linear and pair counts grow a lot, while the binary count grows by about 4.
- **The doubling test:** going from 1,000 to 2,000 numbers, linear search work doubles (2.0), all-pairs work quadruples (4.0), and binary search adds exactly one step.
- **The definition:** $3n^2+10n+5$ stays under $4n^2$ from $n=11$ on (checked to 100,000) and is *over* it at $n=10$. It is never below $3n^2$ at $n=1000$, so $c=3$ does not work. Divided by $n$ it keeps growing (40, 310, 3,010, 30,010), so it is not $O(n)$. Divided by $n^2$ it settles near 3.
- **Two ways to find a pair:** with no matching pair among 2,000 numbers, the all-pairs method took 1,999,000 steps, and the set method took 2,000. Both were counted in steps, and I did not time them. (The set method assumes a set lookup takes about one step, which is usually true and is explained in [[Hash Tables]].)
- **Exponential:** the naive Fibonacci recursion took 177 calls for $n=10$, 21,891 for $n=20$ and 242,785 for $n=25$. Adding 5 to $n$ multiplied the work by about 11. Run on my machine (not part of the code above), $n=32$ took about 1.0 seconds with 7,049,155 calls. Because the calls grow by about 1.618 for each step of $n$, I estimate (by extrapolating, not by running it) that $n=40$ needs 331,160,281 calls and about 50 seconds, and $n=50$ needs about 40.7 billion calls and roughly 1.6 hours. The call counts follow the formula $2F(n+1)-1$, which matches the three counts the code printed (here $F$ is the Fibonacci number). The timings are rough and depend on the machine.
- **Doubling storage:** 1,000 appends cost 1,023 copies in total, about 1 per append, and one million appends cost about 1.05 per append. So each append is cheap *on average*, even though a few of them copy a lot.
- **Limits:** I counted steps in simple model code. Real programs also depend on memory caches, the language and the machine, so real timings can disagree with step counts for small inputs.

## 🔴 4. Deep / formal

**The definition.** We write $f(n)=O(g(n))$ ("$f$ is big-O of $g$") when $f$ is eventually at most a fixed multiple of $g$:

$$f(n)=O\big(g(n)\big)\iff \exists\,c>0,\ n_0\ \text{such that}\ f(n)\le c\,g(n)\ \text{for all}\ n\ge n_0$$

**Where:**
- $f(n)$ is the thing being measured, usually the number of steps an algorithm takes on an input of size $n$. **Matters:** it can be a messy formula with several terms.
- $g(n)$ is the simple comparison shape ($1$, $\log n$, $n$, $n^2$, ...). **Matters:** it is the label you end up using.
- $c$ is a constant multiplier, which you may choose. **Matters:** it is the reason the notation ignores "twice as fast" machines or languages. A constant factor cannot change the class.
- $n_0$ is a starting point, which you may also choose. **Matters:** it allows $f$ to misbehave for small inputs, since only large $n$ counts. Without it, many sensible statements would fail because of the first few cases.
- $\exists$ means "there exist" and $\iff$ means "if and only if". **Matters:** to *prove* $f=O(g)$, you only need to find one $c$ and one $n_0$ that work. To show it is false, you have to show that none can.
- Big-O gives an *upper* bound, so $n=O(n^2)$ is also true. People normally mean the tightest shape they can prove (a lower bound is written $\Omega$ and a matching two-sided bound is written $\Theta$, as defined by Knuth; see the history point below).

![[big-o-definition.png]]
*The definition in a picture. The red curve is $f(x)$ and the blue curve is $c\,g(x)$. They cross and tangle for small $x$, but after the point marked $x_0$ the blue curve stays above the red one for ever. That "after some point, always above" is exactly what $f=O(g)$ asks. Image: Fede Reghe, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Big-O-notation.png), public domain.*

**Example (tiny numbers):** Show that $f(n)=3n^2+10n+5$ is $O(n^2)$.
1. Try $c=4$ and ask when $3n^2+10n+5\le4n^2$. That is the same as $n^2\ge10n+5$.
2. At $n=10$: $100\ge105$ is false, since $f(10)=405>400$.
3. At $n=11$: $121\ge115$ is true, since $f(11)=478\le484=4\cdot11^2$. For larger $n$ the gap only widens, because $n^2$ grows faster than $10n+5$.
4. So $c=4$, $n_0=11$ works. The code checked every $n$ from 11 to 100,000 and the failure at 10.
5. Meaning: the cost formula looks complicated, but for large inputs it behaves like a plain multiple of $n^2$, so it is called "order $n^2$". With $c=3$ it would fail, because the extra $10n+5$ is always above $3n^2$. The constant $c$ must be a little bigger than the leading coefficient.

**Counting all pairs.** Comparing every item with every other item in a list of $n$ items takes

$$\binom{n}{2}=\frac{n(n-1)}{2}\ \text{comparisons}=O(n^2)$$

**Where:**
- $n$ is the number of items. **Matters:** it is the input size that the growth is measured against.
- $\binom{n}{2}$ ("n choose 2") is the number of ways to pick 2 items out of $n$, ignoring order. **Matters:** each pair is compared once, not twice, so you divide by 2.
- $n(n-1)$ counts ordered pairs: $n$ choices for the first item and $n-1$ for the second. **Matters:** the "$-1$" is because an item is not compared with itself.
- Dividing by 2 and dropping the $-1$ leaves $\tfrac12n^2-\tfrac12n$. **Matters:** the leading part is $\tfrac12n^2$. Big-O drops the $\tfrac12$ and the lower term, leaving $O(n^2)$.

**Example (tiny numbers):**
1. $n=5$: $5\cdot4/2=10$ pairs.
2. $n=30$ (the handshake class): $30\cdot29/2=435$. Doubled to $n=60$: $60\cdot59/2=1{,}770$, about 4.07 times as many.
3. $n=1000$: $1000\cdot999/2=499{,}500$. $n=2000$: $1{,}999{,}000$. The code counts these exactly.
4. Meaning: doubling the input multiplies the work by about 4 ($2^2$). Attention compares every token with every other token, so a text twice as long costs about four times as many comparisons (see [[Self-Attention]]).

**Halving gives logarithms.** Binary search on a sorted list of $n$ items throws away half of what is left at every step, so in the worst case it needs

$$T(n)=\lfloor\log_2 n\rfloor+1\ \text{steps}=O(\log n)$$

**Where:**
- $\log_2 n$ is the number of times you can halve $n$ before reaching 1. **Matters:** it grows extremely slowly: doubling $n$ adds only 1.
- $\lfloor\cdot\rfloor$ ("floor") rounds down to a whole number. **Matters:** steps are whole numbers, so the exact count is a whole number.
- The $+1$ counts the last look, when only one item is left. **Matters:** without it the formula would be short by one in the worst case.
- The base of the logarithm does not matter for Big-O, since $\log_2n$ and $\log_{10}n$ differ by a constant factor ($\log_2n\approx3.32\log_{10}n$). **Matters:** that is why people write $O(\log n)$ with no base.

**Example (tiny numbers):**
1. $n=1000$: $\log_2 1000=9.97$, so $\lfloor9.97\rfloor+1=10$ steps. The code found 10 as the longest search, over every possible target.
2. $n=10^6$: $\log_2 10^6=19.93$, so $19+1=20$ steps. A million items, 20 looks.
3. $n=10^{12}$ (a million million): $\log_2\approx39.9$, so about 40 steps.
4. Doubling $n$ from 1,000 to 2,000 raises the count from 10 to 11. The code printed "binary adds 1".
5. Meaning: a method that halves the problem each step is so cheap that even absurdly large inputs take only a few dozen steps. Compare the 1,000,000 steps a plain scan could need.

**Other points (the first two were checked online; the last two are general statements that I did not test):**
- **History (checked against Wikipedia's article on Big O notation):** Paul Bachmann proposed the notation in 1894, Edmund Landau extended it in 1909 (and the little-$o$ notation is credited to Landau), and Hardy introduced $\asymp$ for "same order". Hardy and Littlewood had already defined $\Omega$ in number theory, in a different sense. Donald Knuth (1976) defined $\Omega$ and $\Theta$ in the way computer scientists use them today: $f=\Omega(g)$ means $g=O(f)$, and $\Theta$ is the same-order relation. So Knuth *redefined and popularised* these two symbols for computer science. He did not invent them. I did not read the primary sources, and the web sources differ slightly on the details of who deserves credit.
- **Sorting lower bound (checked against Wikipedia's article on comparison sorts):** any sort that works only by comparing items needs at least $\lceil\log_2(n!)
ceil$ comparisons in the worst case, because $n!$ orderings are possible and each comparison has two outcomes, so $2^{k}\ge n!$ for $k$ comparisons. And $\log_2(n!)=\Theta(n\log n)$, so the bound is $\Omega(n\log n)$, which merge sort meets. The exact count is not always achievable. The article gives the example $n=13$: $\lceil\log_2 13!
ceil=33$ but 34 comparisons are needed. I did not verify that example myself. The "best known matrix-multiplication exponent" is a research result that I did not check and did not state in this note.
- Practical performance depends on caches, memory layout and the actual data. A "slower" shape can win for small or typical inputs, so measure before optimising.
- The classes of hard problems (P, NP and the like) are outside this note.

**What came before, and how this improved on it:**
- **Before:** time a program with a stopwatch and compare.
- **Its limit:** the time depends on the machine, the language and the data, and says nothing about what happens when the input grows.
- **What changed:** count steps as a function of the input size $n$ and keep only how fast the count grows. The notation comes from number theory (Bachmann 1894 and Landau; from memory, not checked) and was brought into the study of algorithms later (Knuth, 1970s; from memory).
- **How much better:** in the code below, when $n$ grows 1,000 times (1,000 to 1,000,000), "look at each item" takes 1,000 times more steps, "halve each time" takes 2 times more (10 to 20 steps), and "compare every pair" takes 1,001,000 times more. The counts are exact and do not depend on the machine.

**Check it yourself:**

```python
import math
print("steps to finish a job on n items (counted, not timed):")
print(f"{'n':>10} {'look at each item':>20} {'halve each time':>17} {'compare every pair':>20}")
for n in (1_000, 1_000_000):
    print(f"{n:>10,} {n:>20,} {math.ceil(math.log2(n)):>17} {n * (n - 1) // 2:>20,}")
print("n grows 1,000 times: first grows", 1_000, "times, second", math.ceil(math.log2(10**6)) // math.ceil(math.log2(10**3)), "times, third", f"{(10**6 * (10**6 - 1) // 2) // (10**3 * (10**3 - 1) // 2):,}", "times")
```

## Related
[[Python]] · [[NumPy]] · [[Pandas]] · [[Self-Attention]] · [[FlashAttention]] · [[Matrix]] · [[GPU]] · [[k-NN]] · [[Vector Database]] · [[Hash Tables]] · [[Binary Search]] · [[Sorting]] · [[Recursion]] · [[Dynamic Programming]] · [[Arrays and Lists]] · [[Stacks and Queues]] · [[Graphs]] · [[Heaps and Priority Queues]] · [[Dijkstra's Algorithm]] · [[Greedy Algorithms]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*. The asymptotic-notation chapter is Chapter 3: called "Growth of Functions" in the second and third editions, and renamed "Characterizing Running Times" in the fourth (checked against a web table of contents and the MIT Press transition guide; I did not open the book).
- Knuth, "Big Omicron and big Omega and big Theta", SIGACT News, 1976 (title and year confirmed by web search; I did not open the paper). Background on the history: [Wikipedia, Big O notation](https://en.wikipedia.org/wiki/Big_O_notation); sorting bound: [Wikipedia, Comparison sort](https://en.wikipedia.org/wiki/Comparison_sort).
- Figures: [Comparison_computational_complexity.svg](https://commons.wikimedia.org/wiki/File:Comparison_computational_complexity.svg) (Cmglee, CC BY-SA 4.0) and [Big-O-notation.png](https://commons.wikimedia.org/wiki/File:Big-O-notation.png) (Fede Reghe, public domain), both on Wikimedia Commons. I viewed both before writing the captions.
- The code and every Example number were run for this note (Python 3.9, standard library only). Steps were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/complexity Big-O:::A way to say how the number of steps grows when the input size $n$ grows. Keep only the biggest part and drop constants.

#flashcards/algorithms/concept #flashcards/topic/complexity List the steps of finding a Big-O class.::Pick the input size $n$, count the basic steps, ask how the count grows as $n$ grows, keep only the biggest part (drop smaller parts and the constant multiplier), then name the shape.

#flashcards/algorithms/concept #flashcards/topic/complexity Name the common Big-O shapes from slowest to fastest growing.::$O(1)$ constant, $O(\log n)$, $O(n)$, $O(n\log n)$, $O(n^2)$, $O(2^n)$ (explodes).

#flashcards/algorithms/concept #flashcards/topic/complexity Why does a slower-growing shape win in the end?::For large inputs the shape matters more than constants, even if it looks slower on tiny inputs.

#flashcards/algorithms/formula #flashcards/topic/complexity Write the formal definition of Big-O and explain each symbol.
?
$$f(n)=O\big(g(n)\big)\iff \exists\,c>0,\ n_0\ \text{such that}\ f(n)\le c\,g(n)\ \text{for all}\ n\ge n_0$$
- $f(n)$: the thing being measured, usually the number of steps on an input of size $n$. It can be a messy formula with several terms.
- $g(n)$: the simple comparison shape ($1$, $\log n$, $n$, $n^2$, ...), the label you end up using.
- $c$: a constant multiplier you may choose. This is why the notation ignores "twice as fast" machines or languages. A constant factor cannot change the class.
- $n_0$: a starting point you may also choose. It lets $f$ be messy for small $n$.
- "After some point, always below $c\,g$" is exactly what $f=O(g)$ asks.

#flashcards/algorithms/example #flashcards/topic/complexity Show that $f(n)=3n^2+10n+5$ is $O(n^2)$.::Try $c=4$: need $3n^2+10n+5\le4n^2$, i.e. $n^2\ge10n+5$. At $n=10$: $f(10)=405>400$, fails. At $n=11$: $f(11)=478\le484=4\cdot11^2$, works, and the gap only widens. So $c=4$, $n_0=11$ works. The code checked every $n$ from 11 to 100,000 and the failure at 10.

#flashcards/algorithms/formula #flashcards/topic/complexity Write the number of pair comparisons among $n$ items and explain each symbol.
?
$$\binom{n}{2}=\frac{n(n-1)}{2}\ \text{comparisons}=O(n^2)$$
- $n$: the number of items.
- $\binom{n}{2}$ ("n choose 2"): the number of ways to pick 2 items out of $n$, ignoring order. Each pair is compared once, so divide by 2.
- $n(n-1)$: ordered pairs, $n$ choices for the first and $n-1$ for the second. The $-1$ is because an item is not compared with itself.
- Dividing by 2 and dropping the $-1$ leaves $\tfrac12n^2-\tfrac12n$. The leading part is $\tfrac12n^2$, and Big-O drops the $\tfrac12$.

#flashcards/algorithms/example #flashcards/topic/complexity How many pairs for $n=5$, 30, 60, 1000 and 2000?::$n=5$: 10. $n=30$: $30\cdot29/2=435$. $n=60$: 1,770, about 4.07 times as many. $n=1000$: 499,500. $n=2000$: 1,999,000. Doubling the input multiplies the work by about 4. Attention compares every token with every other, so a text twice as long costs about four times as many comparisons.

#flashcards/algorithms/formula #flashcards/topic/complexity Write the number of steps of halving search and explain each symbol.
?
$$T(n)=\lfloor\log_2 n\rfloor+1\ \text{steps}=O(\log n)$$
- $\log_2 n$: the number of times you can halve $n$ before reaching 1. It grows extremely slowly: doubling $n$ adds only 1.
- $\lfloor\cdot\rfloor$ (floor): rounds down, because steps are whole numbers.
- The $+1$: counts the last look, when only one item is left. Without it the formula would be short by one in the worst case.
- The base of the logarithm does not matter for Big-O, since different bases differ by a constant factor.

#flashcards/algorithms/example #flashcards/topic/complexity How many steps for $n=1000$, $10^6$ and $10^{12}$?::$n=1000$: $\lfloor9.97\rfloor+1=10$ (the code found 10 as the longest search). $n=10^6$: $19+1=20$. $n=10^{12}$: about 40. Doubling $n$ from 1,000 to 2,000 raises the count from 10 to 11.

#flashcards/algorithms/image #flashcards/topic/complexity ![[big-o-growth-comparison.svg]]
What does this picture show?
?
How the number of steps $N$ grows as input size $n$ grows from 0 to 100, for the common shapes with every constant set to 1. The flat line is 1. Slightly above are $\log_2 n$ and $\sqrt n$, then the straight line $n$, then $n\log_2 n$ climbing steeply. The curves $n^2$, $2^n$ and $n!$ shoot off the top almost at once. (From the note's caption: Cmglee, Wikimedia Commons.)

#flashcards/algorithms/image #flashcards/topic/complexity ![[big-o-definition.png]]
What does this picture show?
?
The definition in a picture. The red curve is $f(x)$ and the blue curve is $c\,g(x)$. They cross and tangle for small $x$, but after the point marked $x_0$ the blue curve stays above the red one forever. That "after some point, always above" is what $f=O(g)$ asks. (From the note's caption: Fede Reghe, Wikimedia Commons, public domain.)
