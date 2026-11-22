---
tags: [algorithms, dynamic-programming, recursion, optimization, programming, python]
status: growing
level: 4
tier: 2
---

# Dynamic Programming

Dynamic programming (DP) is a way to solve a big problem by **solving each small piece once, writing the answer down, and building the big answer from the written-down pieces**. It is used when a plain [[Recursion]] would keep solving the same small problems again and again. The name is historical and says little about the method.

## 🟢 1. Simple (age 12)
Suppose you want to climb a staircase and you can take one or two steps at a time. "How many ways to reach step 10?" is hard to answer in one go. But if you already know the number of ways to reach step 8 and step 9, you just add them. So you write down the answer for step 1, then step 2, then step 3, and so on. Each new answer takes one quick addition, because the earlier answers are on your paper. You never work out the same thing twice.

**Example:** You are packing a small bag for a trip. For each toy you ask, "is it worth the space?" It is hard to decide for all toys at once. So you make a little table: what is the best bag I can pack from just the first toy, for each amount of room? Then from the first two toys? And so on. Each row of the table uses the row above it. The last row has your answer.

## 🔵 2. How it works
A problem is a good fit for DP when it has two features: (a) the answer is built from answers to **smaller versions** of the same problem, and (b) those smaller versions **overlap**, so the same ones are needed many times.

The steps:
1. **Say what a small problem is.** For example, "the best value using the first $i$ items with room $w$". This is called the *state*.
2. **Write the rule** that builds a bigger answer from smaller ones (the *recurrence*). Usually it is a choice between a few options, and you keep the best.
3. **Name the starting answers** (the *base cases*), like "with no items the best value is 0".
4. **Fill in the answers** in an order where the smaller ones are always ready first. Either work *bottom-up* (fill a table from the small cases) or *top-down* (use the recursion, but write each answer down the first time, which is *memoisation*).
5. **Read the answer** from the cell for the full problem. If you also need *which choices* were made, walk back through the table.

![[knapsack-dynamic-programming.gif]]
*The 0/1 knapsack problem solved by dynamic programming. The left table lists 4 items (value $v$, weight $w$) and a capacity of 6. The right table fills in, row by row, the best value for each amount of room, using only the first $i$ items. The last cell (bottom right) is the answer, 9. This is an animation: some viewers show only the first frame. Image: RDSEED, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Knapsack_problem_dynamic_programming.gif), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Two features to check:** *optimal substructure* (the best answer to the big problem is made from best answers to smaller ones) and *overlapping sub-problems* (the same small problems come up repeatedly). Without overlap, memoising saves nothing, and plain divide and conquer (see [[Sorting]]) is enough.
- **Top-down and bottom-up:** top-down is the recursion plus a memory (`functools.lru_cache`). It solves only the sub-problems that are actually reached. Bottom-up fills the whole table in a loop. It avoids deep recursion (Python stops at 1,000 calls deep by default, see [[Recursion]]) and is usually a little faster. Both give the same answer. In the code, the top-down knapsack solved 21 of the 35 cells.
- **Cost:** the work is (number of states) × (work per state), see [[Big-O]]. Knapsack has $(n+1)(W+1)$ states with a constant amount of work each. Edit distance has $(m+1)(n+1)$ states.
- **A warning about "polynomial":** the knapsack table depends on the *number* $W$, not on how many digits it has. A capacity of one million with 100 items needs about 101 million cells (the code counts them). This is called *pseudo-polynomial* time. Knapsack is believed to be hard in general (it is NP-hard, from memory, not checked here), so there is no contradiction.
- **Greedy against DP:** a greedy rule such as "take the best value per kilogram first" is fast but can be wrong. In the code, with items (60, 10), (100, 20), (120, 30) and capacity 50, greedy collects 160 and DP finds 220.
- **Three classic problems:** *knapsack* (choose items under a limit), *edit distance* (fewest changes between two strings, used in spell checkers), and *shortest paths and values over time* (the Bellman equation).
- **Edit distance:** from "kitten" to "sitting" takes 3 edits (swap k for s, swap e for i, add g). The table has 56 cells. The plain recursion needs 29,737 calls for the same answer.
- **Reinforcement learning:** the Bellman equation in [[Q-Learning]] is a dynamic-programming recurrence: the value of a state is built from the values of the states you can reach next. The code runs a tiny corridor and the values settle at 1, 0.9, 0.81, 0.729. Q-learning learns the same kind of values from experience instead of from a known table.
- **Limits:** the code runs small cases. It checks knapsack against brute force on 300 random small cases and edit distance on a handful of strings. I did not time anything, and I did not cover more advanced topics (such as bitmask DP, or saving memory by keeping only one row).

The code covers five things: (1) the knapsack table from the picture, the items taken, and a brute-force check, (2) greedy failing where DP works, (3) edit distance and the plain recursion's call count, (4) top-down against bottom-up, (5) the Bellman equation on a corridor, and how big the table gets.

```python
import functools, itertools, random
# 1) 0/1 knapsack: V[i][w] = best value using the first i items with room w (the table in the picture)
def knapsack(items, W):
    V = [[0] * (W + 1) for _ in range(len(items) + 1)]
    for i, (v, wt) in enumerate(items, 1):
        for w in range(W + 1):
            V[i][w] = V[i - 1][w]                                   # skip item i
            if wt <= w: V[i][w] = max(V[i][w], v + V[i - 1][w - wt])   # or take it
    return V
items = [(5, 4), (4, 3), (3, 2), (2, 1)]                           # (value, weight), capacity 6, as in the picture
V = knapsack(items, 6)
for row in V: print(row)
def chosen(V, items, W):                                            # walk back through the table to see which items were taken
    out = []
    for i in range(len(items), 0, -1):
        if V[i][W] != V[i - 1][W]: out.append(i); W -= items[i - 1][1]
    return sorted(out)
print('best value', V[4][6], '| items taken', chosen(V, items, 6), '| table cells', (len(items) + 1) * 7)
def brute(items, W):
    return max(sum(v for v, _ in c) for r in range(len(items) + 1) for c in itertools.combinations(items, r) if sum(w for _, w in c) <= W)
print('brute force over all', 2 ** len(items), 'subsets:', brute(items, 6))
random.seed(0); ok = 0
for _ in range(300):
    it = [(random.randint(1, 20), random.randint(1, 8)) for _ in range(random.randint(1, 8))]; W = random.randint(0, 20)
    ok += knapsack(it, W)[-1][W] == brute(it, W)
print('random cases where table equals brute force:', ok, 'of 300')
# greedy by value-per-weight is NOT always right
g = [(60, 10), (100, 20), (120, 30)]; W = 50
left, tot = W, 0
for v, wt in sorted(g, key=lambda x: -x[0] / x[1]):
    if wt <= left: left -= wt; tot += v
print('greedy', tot, '| table', knapsack(g, W)[-1][W])
# 2) edit distance: D[i][j] = fewest single-letter edits to turn the first i letters of a into the first j letters of b
def edit(a, b):
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): D[i][0] = i
    for j in range(len(b) + 1): D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return D
D = edit('kitten', 'sitting')
print('kitten -> sitting:', D[-1][-1], '| table cells', len(D) * len(D[0]))
calls = {'n': 0}
def edit_naive(a, b):
    calls['n'] += 1
    if not a: return len(b)
    if not b: return len(a)
    return min(edit_naive(a[:-1], b) + 1, edit_naive(a, b[:-1]) + 1, edit_naive(a[:-1], b[:-1]) + (a[-1] != b[-1]))
print('naive recursion gives', edit_naive('kitten', 'sitting'), 'using', calls['n'], 'calls')
print('tiny cases:', edit('ab', 'b')[-1][-1], edit('', 'abc')[-1][-1], edit('same', 'same')[-1][-1], edit('abc', 'xyz')[-1][-1])
print('table row for the last letters: D[6][7] =', D[6][7], 'from D[5][7] =', D[5][7], ', D[6][6] =', D[6][6], ', D[5][6] =', D[5][6])
# 3) top-down (remember answers as you go) reaches the same answer but only solves the sub-problems it needs
runs = {'n': 0}
@functools.lru_cache(maxsize=None)
def kn_top(i, w):
    runs['n'] += 1
    if i == 0: return 0
    v, wt = items[i - 1]
    best = kn_top(i - 1, w)
    if wt <= w: best = max(best, v + kn_top(i - 1, w - wt))
    return best
print('top-down:', kn_top(4, 6), '| sub-problems actually solved', runs['n'], 'of', 5 * 7, 'table cells')
# 4) Bellman equation on a tiny corridor: states 0..4, goal is state 4 (reward 1 for stepping into it), discount 0.9
gamma = 0.9; Vs = [0.0] * 5
for sweep in range(1, 8):
    new = Vs[:]
    for s in range(4):
        new[s] = max((1.0 if s2 == 4 else 0.0) + gamma * Vs[s2] for s2 in (max(s - 1, 0), s + 1))
    Vs = new
    if sweep in (1, 2, 3, 4, 7): print('sweep', sweep, [round(x, 3) for x in Vs])
print('exact: 1, 0.9, 0.81, 0.729 for states 3, 2, 1, 0:', [round(gamma ** k, 3) for k in range(4)])
# 5) how the table grows: cells = (n + 1)(W + 1); a big capacity makes it big
for n, W in ((4, 6), (100, 1000), (100, 10 ** 6)): print('items', n, 'capacity', W, '-> cells', (n + 1) * (W + 1))
# [0, 0, 0, 0, 0, 0, 0]
# [0, 0, 0, 0, 5, 5, 5]
# [0, 0, 0, 4, 5, 5, 5]
# [0, 0, 3, 4, 5, 7, 8]
# [0, 2, 3, 5, 6, 7, 9]
# best value 9 | items taken [2, 3, 4] | table cells 35
# brute force over all 16 subsets: 9
# random cases where table equals brute force: 300 of 300
# greedy 160 | table 220
# kitten -> sitting: 3 | table cells 56
# naive recursion gives 3 using 29737 calls
# tiny cases: 1 3 0 3
# table row for the last letters: D[6][7] = 3 from D[5][7] = 4 , D[6][6] = 2 , D[5][6] = 3
# top-down: 9 | sub-problems actually solved 21 of 35 table cells
# sweep 1 [0.0, 0.0, 0.0, 1.0, 0.0]
# sweep 2 [0.0, 0.0, 0.9, 1.0, 0.0]
# sweep 3 [0.0, 0.81, 0.9, 1.0, 0.0]
# sweep 4 [0.729, 0.81, 0.9, 1.0, 0.0]
# sweep 7 [0.729, 0.81, 0.9, 1.0, 0.0]
# exact: 1, 0.9, 0.81, 0.729 for states 3, 2, 1, 0: [1.0, 0.9, 0.81, 0.729]
# items 4 capacity 6 -> cells 35
# items 100 capacity 1000 -> cells 101101
# items 100 capacity 1000000 -> cells 101000101
```
Reading the output (Python 3.9, standard library only):
- **Knapsack table:** the five printed rows are exactly the table in the picture (last row `0 2 3 5 6 7 9`). The best value is 9, using items 2, 3 and 4, and the table has 35 cells (5 rows × 7 columns). Brute force over all 16 subsets also gives 9, and the table matched brute force on all 300 random small cases.
- **Greedy fails:** greedy by value-per-weight gets 160, the table gets 220.
- **Edit distance:** "kitten" to "sitting" is 3, from a table of 56 cells. The plain recursion gives 3 as well, but uses 29,737 calls. Small checks: "ab" to "b" is 1, "" to "abc" is 3, "same" to "same" is 0, "abc" to "xyz" is 3. The last cell comes from $\min(4+1,\ 2+1,\ 3+1)=3$.
- **Top-down:** the memoised version also gives 9, and solves 21 of the 35 cells (only those it needs).
- **Bellman corridor:** the values spread backwards from the goal, one state per sweep, and settle at 0.729, 0.81, 0.9, 1.0 for states 0 to 3 (the goal state, 4, has value 0 here). They equal $0.9^3, 0.9^2, 0.9^1, 0.9^0$.
- **Table size:** 4 items with capacity 6 give 35 cells, 100 items with capacity 1,000 give 101,101, and with capacity 1,000,000 give 101,000,101.
- **Limits:** operations counted, nothing timed. Small examples only.

## 🔴 4. Deep / formal

**The knapsack recurrence.** Let $V(i,w)$ be the best total value using only the first $i$ items when the bag holds weight at most $w$. For item $i$ with value $v_i$ and weight $w_i$:

$$V(0,w)=0,\qquad V(i,w)=\begin{cases}V(i-1,w) & w_i>w\\ \max\big(V(i-1,w),\ v_i+V(i-1,w-w_i)\big) & w_i\le w\end{cases}$$

**Where:**
- $V(i,w)$ is the best value from the first $i$ items with room $w$. **Matters:** this is the "state"; the answer to the whole problem is $V(n,W)$.
- $V(0,w)=0$ is the base case: with no items there is no value. **Matters:** it anchors the table, and every other cell is built from it.
- $V(i-1,w)$ is the option "skip item $i$". **Matters:** it is always allowed, so the best value never decreases as items are added.
- $v_i+V(i-1,w-w_i)$ is the option "take item $i$": you gain its value $v_i$ and use up $w_i$ of the room, so the earlier items must fit in $w-w_i$. **Matters:** this is what makes the choice; without comparing the two options, you would either always take or never take an item.
- The condition $w_i\le w$ says the item fits. **Matters:** without it the formula would use a negative room.
- The recurrence works because the best answer for the first $i$ items either leaves out item $i$ (and is then optimal for $i-1$ items) or includes it (and is then optimal for $i-1$ items in the remaining room). I checked it by brute force on small cases; the general proof is a short exchange argument that I did not write out.

**Example (tiny numbers):** the picture's table, cell $V(4,6)$. Item 4 has $v_4=2$, $w_4=1$.
1. Skip item 4: $V(3,6)=8$.
2. Take item 4: $2+V(3,5)=2+7=9$.
3. $\max(8,9)=9$. The code's last cell is 9.
4. Meaning: the best bag for capacity 6 is worth 9. Walking back (9 differs from 8, so item 4 was taken; then $V(3,5)=7$ differs from $V(2,5)=5$, so item 3 was taken; then $V(2,3)=4$ differs from $V(1,3)=0$, so item 2 was taken) gives items 2, 3 and 4, with weight $3+2+1=6$ and value $4+3+2=9$.

**The edit-distance recurrence.** Let $D(i,j)$ be the fewest single-letter changes (insert, delete or replace a letter) needed to turn the first $i$ letters of string $a$ into the first $j$ letters of string $b$.

$$D(i,0)=i,\qquad D(0,j)=j,\qquad D(i,j)=\min\big(D(i-1,j)+1,\ D(i,j-1)+1,\ D(i-1,j-1)+[a_i\ne b_j]\big)$$

**Where:**
- $D(i,0)=i$ and $D(0,j)=j$ are the base cases. **Matters:** turning $i$ letters into nothing takes $i$ deletions, and building $j$ letters from nothing takes $j$ insertions.
- $D(i-1,j)+1$ means delete the last letter of $a$ (then match the rest). **Matters:** it costs one edit.
- $D(i,j-1)+1$ means insert the last letter of $b$. **Matters:** it also costs one edit.
- $D(i-1,j-1)+[a_i\ne b_j]$ means match or replace the last letters. The bracket $[a_i\ne b_j]$ is 1 if the letters differ and 0 if they are the same. **Matters:** equal letters are free, which is why similar words end up close.
- $\min$ picks the cheapest of the three. **Matters:** this is the "best choice" step; each cell uses only its three neighbours (above, left, and diagonal).

**Example (tiny numbers):** the last cell of the "kitten" to "sitting" table, $D(6,7)$.
1. $D(5,7)+1=4+1=5$ (delete the last "n").
2. $D(6,6)+1=2+1=3$ (insert the last "g").
3. $D(5,6)+[\text{n}\ne\text{g}]=3+1=4$ (replace).
4. $\min(5,3,4)=3$. The code got 3.
5. Meaning: three edits are enough and fewer are not possible. Intuitively: replace k with s, replace e with i, insert g.

**The Bellman equation.** In a world with states $s$, the best long-run value of a state is the best immediate reward plus the discounted value of where you land:

$$V(s)=\max_{a}\big[\,r(s,a)+\gamma\,V(s')\,\big]$$

**Where:**
- $V(s)$ is the value of state $s$, the total future reward you can expect if you play well from there. **Matters:** this is the table of answers, one entry per state.
- $\max_a$ means choose the best action $a$. **Matters:** it is the "best choice" step, as in knapsack and edit distance.
- $r(s,a)$ is the reward for taking action $a$ in state $s$. **Matters:** it is the only source of value; without it all values are zero.
- $s'$ is the state you end up in. Here the moves are fixed (no luck), so $s'$ is known. **Matters:** the value of $s'$ is the smaller answer that the bigger one is built from.
- $\gamma$ is the discount, a number between 0 and 1 (0.9 in the code). **Matters:** it makes a reward that is further away worth less, and keeps the total finite.
- With random outcomes, the term $V(s')$ becomes an average over the possible $s'$. I did not cover that here (see [[Q-Learning]]).

**Example (tiny numbers):** the corridor with states 0 to 4, where stepping into state 4 gives reward 1, and $\gamma=0.9$. For state 2:
1. Step right to state 3: $0+0.9\cdot V(3)=0.9\cdot1=0.9$.
2. Step left to state 1: $0+0.9\cdot V(1)=0.9\cdot0.81=0.729$.
3. $\max(0.9,0.729)=0.9$. The code's value for state 2 is 0.9.
4. Meaning: from state 2 the best plan is to walk straight to the goal, and the value shrinks by a factor of 0.9 for each step away from it: 1, 0.9, 0.81, 0.729.

**Other points (from memory, not checked here):**
- The method and its name are credited to Richard Bellman, in the 1950s. I did not check the history, or the story that the name was chosen partly for how it sounded.
- Other well-known DP problems: longest common subsequence, shortest paths (Bellman–Ford, Floyd–Warshall), matrix-chain multiplication, and the Viterbi algorithm for hidden Markov models. I did not cover them here.
- Knapsack being NP-hard, and the idea of pseudo-polynomial time, are standard results that I have not looked up for this note.
- Many problems can be written both ways (top-down or bottom-up). Which is better depends on the problem and the language.

**What came before, and how this improved on it:**
- **Before:** solve a problem by plain recursion, which solves the same small sub-problem again and again.
- **Its limit:** the work can double at every level. Computing Fibonacci numbers by the plain rule makes a number of calls that grows like the answer itself.
- **What changed:** remember each sub-answer the first time. Richard Bellman named the method in the 1950s (from memory, not checked).
- **How much better:** in the code below, `fib(30)` by the plain rule makes 2,692,537 calls, and with a remembered table it makes 59, which is 45,636 times fewer. For `fib(40)` the plain rule would make 331,160,281 calls and the table 79.

**Check it yourself:**

```python
import sys
calls = 0
def naive(n):
    global calls; calls += 1
    return n if n < 2 else naive(n - 1) + naive(n - 2)
memo = {}; mcalls = 0
def fast(n):
    global mcalls; mcalls += 1
    if n in memo: return memo[n]
    memo[n] = n if n < 2 else fast(n - 1) + fast(n - 2)
    return memo[n]
print("fib(30) =", naive(30), " naive calls:", f"{calls:,}")
print("fib(30) =", fast(30), " calls with a remembered table:", mcalls, f"({calls // mcalls:,}x fewer)")
print("naive calls for fib(40), by the same rule 2*fib(n+1)-1:", f"{2 * fast(41) - 1:,}", " memo needs", 2 * 40 - 1)
```

## Related
[[Recursion]] · [[Big-O]] · [[Sorting]] · [[Binary Search]] · [[Hash Tables]] · [[Q-Learning]] · [[Python]] · [[Dijkstra's Algorithm]] · [[Greedy Algorithms]] · [[Git]]
- [[Backtracking]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, chapter on dynamic programming (from memory; I did not open it for this note).
- Python documentation for `functools.lru_cache` (docs.python.org).
- Figure: [Knapsack_problem_dynamic_programming.gif](https://commons.wikimedia.org/wiki/File:Knapsack_problem_dynamic_programming.gif) (RDSEED, CC BY-SA 4.0), on Wikimedia Commons. I viewed the final frame (the full table) before writing the caption. Its table is the same as the one the code prints.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/algorithm-design Dynamic programming (DP):::Build the answer to a big problem from answers to smaller versions of it, writing each answer down so it is computed only once.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design When is a problem a good fit for DP?::When (a) the answer is built from answers to smaller versions of the same problem and (b) those smaller versions overlap, so the same ones are needed many times.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design List the steps of dynamic programming.::Say what a small problem is (the state), write the rule that builds a bigger answer from smaller ones (the recurrence), name the base cases, fill in the answers in an order where smaller ones are ready first, then read the answer from the cell for the full problem.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design Bottom-up against top-down (memoisation)?::Bottom-up fills a table from the small cases. Top-down uses recursion but writes each answer down the first time it is found (memoisation).

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the 0/1 knapsack recurrence and explain each symbol.
?
$$V(0,w)=0,\qquad V(i,w)=\begin{cases}V(i-1,w) & w_i>w\\ \max\big(V(i-1,w),\ v_i+V(i-1,w-w_i)\big) & w_i\le w\end{cases}$$
- $V(i,w)$: the best value from the first $i$ items with room $w$. The answer to the whole problem is $V(n,W)$.
- $V(0,w)=0$: the base case, no items means no value.
- $V(i-1,w)$: the option "skip item $i$". Always allowed, so the best value never decreases as items are added.
- $v_i+V(i-1,w-w_i)$: the option "take item $i$": gain its value $v_i$ and use up $w_i$ of the room, so earlier items must fit in $w-w_i$.

#flashcards/algorithms/example #flashcards/topic/algorithm-design In the picture's table, item 4 has $v_4=2$, $w_4=1$, and $V(3,6)=8$, $V(3,5)=7$. What is $V(4,6)$ and which items were taken?::Skip: 8. Take: $2+7=9$. $\max=9$. Walking back (9 differs from 8, so item 4 taken; $V(3,5)=7$ differs from $V(2,5)=5$, so item 3 taken; $V(2,3)=4$ differs from $V(1,3)=0$, so item 2 taken) gives items 2, 3 and 4, with weight $3+2+1=6$ and value $4+3+2=9$.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the edit-distance recurrence and explain each symbol.
?
$$D(i,0)=i,\qquad D(0,j)=j,\qquad D(i,j)=\min\big(D(i-1,j)+1,\ D(i,j-1)+1,\ D(i-1,j-1)+[a_i\ne b_j]\big)$$
- $D(i,0)=i$ and $D(0,j)=j$: base cases. Turning $i$ letters into nothing takes $i$ deletions, and building $j$ letters from nothing takes $j$ insertions.
- $D(i-1,j)+1$: delete the last letter of $a$ (then match the rest), one edit.
- $D(i,j-1)+1$: insert the last letter of $b$, one edit.
- $D(i-1,j-1)+[a_i\ne b_j]$: match or replace the last letters. The bracket is 1 if the letters differ and 0 if equal, so equal letters are free.

#flashcards/algorithms/example #flashcards/topic/algorithm-design What is $D(6,7)$ for "kitten" to "sitting"?::Delete: $D(5,7)+1=5$. Insert: $D(6,6)+1=3$. Replace: $D(5,6)+1=4$. $\min=3$ (the code got 3). Three edits: replace k with s, replace e with i, insert g.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the value of a state with fixed moves and explain each symbol.
?
$$V(s)=\max_{a}\big[\,r(s,a)+\gamma\,V(s')\,\big]$$
- $V(s)$: the value of state $s$, the total future reward you can expect if you play well. This is the table of answers, one entry per state.
- $\max_a$: choose the best action, the "best choice" step as in knapsack and edit distance.
- $r(s,a)$: the reward for taking action $a$ in state $s$. It is the only source of value. Without it all values are zero.
- $s'$: the state you end up in. Here moves are fixed (no luck), so $s'$ is known.
- $\gamma$: the discount for later rewards.

#flashcards/algorithms/example #flashcards/topic/algorithm-design Corridor of states 0 to 4, stepping into state 4 gives reward 1, $\gamma=0.9$. What is $V(2)$?::Right to state 3: $0.9\cdot V(3)=0.9\cdot1=0.9$. Left to state 1: $0.9\cdot0.81=0.729$. $\max=0.9$ (the code's value). The value shrinks by 0.9 per step away from the goal: 1, 0.9, 0.81, 0.729.

#flashcards/algorithms/image #flashcards/topic/algorithm-design ![[knapsack-dynamic-programming.gif]]
What does this picture show?
?
The 0/1 knapsack problem solved by dynamic programming. The left table lists 4 items (value $v$, weight $w$) and a capacity of 6. The right table fills in, row by row, the best value for each amount of room using only the first $i$ items. The last cell (bottom right) is the answer, 9. It is an animation, and some viewers show only the first frame. (From the note's caption: RDSEED, Wikimedia Commons.)

#flashcards/algorithms/cloze #flashcards/topic/algorithm-design DP needs ==overlapping== smaller problems. The recurrence picks the best of a few options.
