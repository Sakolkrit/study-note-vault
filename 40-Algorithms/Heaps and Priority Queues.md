---
tags: [algorithms, data-structures, heap, priority-queue, sorting, programming, python]
status: growing
level: 4
tier: 2
---

# Heaps and Priority Queues

A **priority queue** is a line where the item with the **highest priority** (or the smallest value) is served first, not the one who arrived first. A **heap** is the standard way to build one: a clever arrangement of items in an ordinary list that always keeps the best item at the front, and can add or remove items quickly. Python's `heapq` module is a heap. It is used for task schedulers, for finding the "top 10" of a huge stream, for sorting (*heapsort*), and as the engine of shortest-path searches in [[Graphs]].

## 🟢 1. Simple (age 12)
Imagine a hospital waiting room. People are not seen in the order they arrive. The most urgent patient is seen first, whatever time they came in. When a new patient arrives, the nurse puts them in the right place, and when the doctor is free, the nurse picks the most urgent one.

A heap is a neat way to keep that waiting room organised, so that finding the most urgent person is instant, and fitting a new person in is quick.

**Example:** A game keeps a list of monsters, and each turn the one with the lowest health acts first. New monsters appear and old ones are removed all the time. A heap always gives you the weakest monster straight away.

## 🔵 2. How it works
1. **Picture a pyramid.** Put the best item at the top. Below it sit two items, below those four, and so on, filling each row from the left. The rule is: **every item is "better or equal" to the two items directly below it.** So the best item of all is always at the top.
2. **Store the pyramid in a plain list.** Read the pyramid row by row, left to right, and write the items into a list. No pointers are needed: a tiny rule tells you where each item's children and parent are (equation 1 below).
3. **To add an item:** put it at the end of the list (the next free spot at the bottom of the pyramid). If it is better than its parent, swap them. Repeat until it is in the right place. This is called *sifting up*, and it climbs at most the height of the pyramid.
4. **To take out the best item:** take the one at the top. Move the *last* item of the list to the top to fill the hole. If it is worse than one of its children, swap it with the better child. Repeat down the pyramid. This is called *sifting down*.
5. **Why it is fast:** a pyramid with a million items is only about 20 rows high, so each add or remove takes about 20 small steps, not a million.

![[binary-heap-indexing.png]]
*The shape of a binary heap with 12 items, numbered row by row starting from 1 (so item 2 and 3 are the children of item 1, items 4 and 5 are the children of item 2, and so on). In Python the list starts at 0, so the numbers in the code are one less (equation 1 below). Image: Nagae, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Binary_heap_indexing.png), CC BY 2.5.*

![[heap-insert-animation.gif]]
*Adding the number 15 to a heap in which the biggest value is kept at the top (a "max-heap"). The new value is put in the next free spot at the bottom, then it "bubbles up" past 8 and then past 11, until it reaches the top. This is an animation: some viewers show only the first frame. I viewed all 5 of its frames, and the code below repeats this exact example. Image: Sanglee123, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:InsertingIntoBinaryHeap.gif), CC0.*

## 🟠 3. Technical detail
- **Operations and their cost** (see [[Big-O]]): finding the best item is $O(1)$. Adding and removing the best are $O(\log n)$, because the pyramid has about $\log_2 n$ rows (see [[Binary Search]] for where $\log_2$ comes from). Building a heap from a whole list at once is $O(n)$ (equation 2 below).
- **The rule is not "sorted":** a heap only orders each parent against its two children. The items at the same level are in no special order, and the list is not sorted. This looser rule is exactly why it is cheap to maintain.
- **Python's `heapq`:** it gives a **min-heap** (the smallest at the front). Use `heappush`, `heappop`, `heapify`, `heapreplace`, and `nlargest` / `nsmallest`. For a max-heap, store negative numbers (the picture's heap, with the biggest at the top, was built in the code with a flipped comparison instead).
- **Priority queue with ties:** store tuples like `(priority, arrival number, item)`. Python compares tuples left to right, so equal priorities are then served in arrival order (the code's job list does this).
- **Heapsort:** build a heap, then take out the smallest again and again. It sorts in $O(n\log n)$ with no extra memory, but it needs about twice as many comparisons as the minimum possible: the code counted 16,797 for 1,000 items against a lower bound of 8,530 (see [[Sorting]]).
- **Top-$k$ of a stream:** to find the 10 biggest of a huge stream, keep a small min-heap of size 10. A new item enters only if it is bigger than the smallest of the ten. The code does this on 100,000 numbers and only about 100 of them ever enter the heap (equation 3 below). That is far cheaper than sorting everything.
- **Where heaps appear:** schedulers (the next job to run), event simulations (the next event in time), shortest-path search in weighted graphs (Dijkstra's algorithm, see [[Dijkstra's Algorithm]]), data compression (Huffman coding), and merging many sorted lists. In ML, picking the best few candidates at each step of beam search or in a nearest-neighbour search (see [[Vector Database]]) is a top-$k$ problem. That is from memory about how those are usually done; I did not check any particular library.
- **Limits:** the code counts swaps and comparisons. It does not time anything, and it does not compare with other priority-queue structures (balanced trees, Fibonacci heaps) or test huge inputs.

The code covers six things: (1) a heap in a list, the picture's example and the index rule, (2) the cost of one push, (3) building a heap in one go against pushing one by one, (4) heapsort, (5) the top-$k$ of a stream, with an exact tiny check, (6) a priority queue of jobs with ties.

```python
import heapq, random, math, itertools
# 1) a binary heap lives in a plain list: the children of position i are 2i+1 and 2i+2, the parent is (i-1)//2 (counting from 0)
class Heap:
    def __init__(self, better=lambda a, b: a < b): self.a, self.better, self.swaps, self.cmps = [], better, 0, 0
    def _up(self, i):
        while i > 0:
            p = (i - 1) // 2; self.cmps += 1
            if not self.better(self.a[i], self.a[p]): break
            self.a[i], self.a[p] = self.a[p], self.a[i]; self.swaps += 1; i = p
    def _down(self, i):
        n = len(self.a)
        while True:
            best = i
            for c in (2 * i + 1, 2 * i + 2):
                if c < n:
                    self.cmps += 1
                    if self.better(self.a[c], self.a[best]): best = c
            if best == i: return
            self.a[i], self.a[best] = self.a[best], self.a[i]; self.swaps += 1; i = best
    def push(self, x): self.a.append(x); self._up(len(self.a) - 1)
    def pop(self):
        top, last = self.a[0], self.a.pop()
        if self.a: self.a[0] = last; self._down(0)
        return top
    def ok(self): return all(not self.better(self.a[i], self.a[(i - 1) // 2]) for i in range(1, len(self.a)))
h = Heap(lambda a, b: a > b); h.a = [11, 5, 8, 3, 4]                        # the max-heap in the picture
h.push(15); print('picture: after pushing 15 the list is', h.a, '| swaps', h.swaps, '| still a heap:', h.ok())
print('index rule: parent of positions 1..6 is', [(i - 1) // 2 for i in range(1, 7)], '| children of 1 are', (3, 4), '=', (2 * 1 + 1, 2 * 1 + 2))
random.seed(3); data = [random.randint(0, 99) for _ in range(1000)]
m = Heap()
for x in data: m.push(x)
out = [m.pop() for _ in range(len(data))]
hq = list(data); heapq.heapify(hq)
print('1000 pushes then 1000 pops come out sorted:', out == sorted(data), '| matches heapq:', out == [heapq.heappop(hq) for _ in range(len(data))])
# 2) cost of a push: at most one swap per level; the number of levels is floor(log2 n)
w = Heap()
for x in range(1000, 0, -1): w.push(x)                                       # each new value is the smallest so far: it climbs all the way up
print('1000 pushes of falling values: swaps', w.swaps, '| sum of floor(log2 k) for k=1..1000 =', sum(int(math.log2(k)) for k in range(1, 1001)), '| levels for n=1000:', int(math.log2(1000)) + 1)
# 3) building a heap from a whole list at once (bottom-up) costs less than n swaps
def build(a):
    b = Heap(); b.a = list(a)
    for i in range(len(a) // 2 - 1, -1, -1): b._down(i)
    return b
for n in (7, 1000):
    worst = max(build(random.sample(range(n * 3), n)).swaps for _ in range(200)); hi = build(range(n, 0, -1)).swaps
    print('n', n, '| swaps (random order, worst of 200):', worst, '| falling order:', hi, '| n - popcount(n) =', n - bin(n).count('1'))
def sum_heights(n):                                                          # height of node i in a heap of n nodes: how far down the longest path goes
    t = 0
    for i in range(n):
        hgt, j = 0, i
        while 2 * j + 1 < n: j = 2 * j + 1; hgt += 1
        t += hgt
    return t
print('sum of node heights equals n - popcount(n) for every n up to 300:', all(sum_heights(n) == n - bin(n).count('1') for n in range(1, 301)))
print('one-by-one pushes of the same 1000 values (random order): swaps', (lambda q: [q.push(x) for x in data] and q.swaps)(Heap()), '| bottom-up build: swaps', build(data).swaps)
# 4) heapsort: build, then pop the smallest again and again
def heapsort(a):
    q = build(a); r = [q.pop() for _ in range(len(a))]; return r, q.cmps
r, c = heapsort(data); print('heapsort sorted correctly:', r == sorted(data), '| comparisons for n=1000:', c, '| lower bound ceil(log2 n!) =', math.ceil(math.lgamma(1001) / math.log(2)))
# 5) the k largest items of a stream, keeping only k in memory
def top_k(stream, k):
    hp, entered = [], 0
    for x in stream:
        if len(hp) < k: heapq.heappush(hp, x); entered += 1
        elif x > hp[0]: heapq.heapreplace(hp, x); entered += 1
    return sorted(hp, reverse=True), entered
stream = random.sample(range(10 ** 6), 100000); got, ent = top_k(stream, 10)
print('top 10 of 100000 correct:', got == sorted(stream, reverse=True)[:10] == heapq.nlargest(10, stream), '| items that entered the heap:', ent)
H = lambda n: sum(1 / i for i in range(1, n + 1))
runs = [top_k(random.sample(range(10 ** 6), 100000), 10)[1] for _ in range(300)]
print('average entries over 300 shuffles:', round(sum(runs) / 300, 1), '| k(1 + H_n - H_k) =', round(10 * (1 + H(100000) - H(10)), 1))
# exact check on a tiny case: all 24 orders of 4 items, k = 2
tot = sum(top_k(p, 2)[1] for p in itertools.permutations(range(4))) / 24
print('n=4, k=2, exact average over all 24 orders:', round(tot, 4), '| formula:', round(2 * (1 + H(4) - H(2)), 4))
# 6) a priority queue of jobs; a counter breaks ties so equal priorities leave in arrival order
jobs = [(2, 'email'), (1, 'fire alarm'), (3, 'lunch'), (1, 'bug fix'), (2, 'report')]
pq = []
for n, (p, name) in enumerate(jobs): heapq.heappush(pq, (p, n, name))
print('served in order:', [heapq.heappop(pq)[2] for _ in range(len(jobs))])
# picture: after pushing 15 the list is [15, 5, 11, 3, 4, 8] | swaps 2 | still a heap: True
# index rule: parent of positions 1..6 is [0, 0, 1, 1, 2, 2] | children of 1 are (3, 4) = (3, 4)
# 1000 pushes then 1000 pops come out sorted: True | matches heapq: True
# 1000 pushes of falling values: swaps 7987 | sum of floor(log2 k) for k=1..1000 = 7987 | levels for n=1000: 10
# n 7 | swaps (random order, worst of 200): 4 | falling order: 4 | n - popcount(n) = 4
# n 1000 | swaps (random order, worst of 200): 784 | falling order: 992 | n - popcount(n) = 994
# sum of node heights equals n - popcount(n) for every n up to 300: True
# one-by-one pushes of the same 1000 values (random order): swaps 1198 | bottom-up build: swaps 718
# heapsort sorted correctly: True | comparisons for n=1000: 16797 | lower bound ceil(log2 n!) = 8530
# top 10 of 100000 correct: True | items that entered the heap: 104
# average entries over 300 shuffles: 101.0 | k(1 + H_n - H_k) = 101.6
# n=4, k=2, exact average over all 24 orders: 3.1667 | formula: 3.1667
# served in order: ['fire alarm', 'bug fix', 'email', 'report', 'lunch']
```
Reading the output (Python 3.9, standard library only):
- **Picture:** pushing 15 into `[11, 5, 8, 3, 4]` (biggest at the top) gives `[15, 5, 11, 3, 4, 8]` with 2 swaps, as in the animation. The parents of positions 1 to 6 are 0, 0, 1, 1, 2, 2.
- **Sorting by popping:** 1,000 pushes followed by 1,000 pops give the sorted list, and the same as `heapq`.
- **Cost of a push:** pushing 1,000 falling values (each climbs to the top) took 7,987 swaps, exactly the sum of $\lfloor\log_2k\rfloor$ for $k=1$ to 1,000. A heap of 1,000 items has 10 rows.
- **Building at once:** for $n=7$, the bound $n-\text{popcount}(n)$ is 4 and the swaps were 4. For $n=1000$ the bound is 994, falling input took 992 swaps and random input at most 784. The sum of node heights equals $n-\text{popcount}(n)$ for every $n$ up to 300. Pushing the same 1,000 random values one by one took 1,198 swaps, and building at once took 718.
- **Heapsort:** correct, with 16,797 comparisons against the lower bound of 8,530.
- **Top-10 of 100,000:** correct (it matches a full sort and `heapq.nlargest`), and 104 items entered the heap. Averaged over 300 shuffles, 101.0 entered, against the formula's 101.6. On a tiny case (all 24 orders of 4 items, $k=2$) the exact average is 3.1667, the same as the formula.
- **Jobs:** the fire alarm, then the bug fix (same priority, arrived earlier than the other priority-1 job), then the email and the report, and lunch last.
- **Limits:** operation counts only, nothing timed.

## 🔴 4. Deep / formal

**Where the relatives are.** With positions counted from 0, the item at position $i$ has its two children at positions $2i+1$ and $2i+2$, and its parent at:

$$\text{left}(i)=2i+1,\qquad \text{right}(i)=2i+2,\qquad \text{parent}(i)=\left\lfloor\frac{i-1}{2}\right\rfloor$$

**Where:**
- $i$ is a position in the list (0 is the top). **Matters:** these rules are all a heap needs to act like a tree, so it uses no pointers and no extra memory.
- $2i+1$ and $2i+2$ are the positions of the left and right child. **Matters:** row $r$ of the pyramid holds positions $2^r-1$ to $2^{r+1}-2$, so doubling $i$ lands on the next row.
- $\lfloor\cdot\rfloor$ means "round down". **Matters:** the two children $2i+1$ and $2i+2$ must both map back to $i$: $\lfloor 2i/2\rfloor=i$ and $\lfloor(2i+1)/2\rfloor=i$.
- If you count from 1 instead of 0 (as the picture does), the rules become $2i$, $2i+1$ and $\lfloor i/2\rfloor$. **Matters:** mixing the two conventions is a classic off-by-one bug.

**Example (tiny numbers):** the list after the picture's push, `[15, 5, 11, 3, 4, 8]`.
1. The item at position 5 is 8. Its parent is $\lfloor(5-1)/2\rfloor=2$, which holds 11. In the picture 8 is indeed the child of 11.
2. The children of position 1 (which holds 5) are $2\cdot1+1=3$ and $2\cdot1+2=4$, which hold 3 and 4. The code printed children (3, 4).
3. Meaning: every parent is at least as big as its children (15 above 5 and 11, 11 above 8, 5 above 3 and 4), and the whole tree is found by arithmetic on list positions.

**Building a heap in one go.** To build a heap from $n$ items already in a list, go through the positions from the middle back to the top and sift each one down. An item at height $h$ (the longest downward path below it) can need at most $h$ swaps. So the total number of swaps is at most the sum of all the heights:

$$\text{swaps}\ \le\ \sum_{v}h(v)\ =\ n-\operatorname{popcount}(n)\ <\ n$$

**Where:**
- $h(v)$ is the height of position $v$: how many steps down the longest path from it goes. **Matters:** half of the items are leaves (height 0) and need no work, a quarter have height 1, an eighth have height 2, and so on. The cheap items are the many; the costly ones are the few.
- $\sum_v h(v)$ adds up the worst case for every position. **Matters:** it is the most the build can ever swap, whatever the input order.
- $\operatorname{popcount}(n)$ is the number of 1s when $n$ is written in binary. **Matters:** the sum of heights equals $n-\operatorname{popcount}(n)$. I checked this by running the code for every $n$ up to 300; the general proof is a short counting argument that I did not write out.
- The result, below $n$, means building is $O(n)$. **Matters:** pushing $n$ items one at a time is $O(n\log n)$, which is more work: the code counted 1,198 swaps for the one-by-one way and 718 for building at once, on the same 1,000 numbers.

**Example (tiny numbers):** $n=7$ (a full pyramid of 3 rows).
1. Heights: the top has 2, the two items in the middle row have 1 each, and the four leaves have 0. The sum is $2+1+1+0+0+0+0=4$.
2. The formula gives $7-\operatorname{popcount}(7)=7-3=4$ ($7$ is `111` in binary).
3. The code's swaps for 7 items were 4 for the worst order it tried. For $n=1000$ the bound is 994 and the falling-order input needed 992.
4. Meaning: the work to build a heap is at most about one swap per item, however many items there are.

**Top-$k$ from a stream.** Keep the $k$ biggest items seen so far in a min-heap. The first $k$ items always enter. After that, the $i$-th item is among the $k$ biggest of the first $i$ with probability $k/i$ (if the order is random), and only then does it enter the heap. So the expected number of items that enter the heap is:

$$\mathbb E[\text{entries}]=k+\sum_{i=k+1}^{n}\frac{k}{i}=k\,\big(1+H_n-H_k\big)\ \approx\ k\,\Big(1+\ln\frac nk\Big)$$

**Where:**
- $n$ is the number of items in the stream and $k$ is how many you keep. **Matters:** the number of entries depends on $k$ and the *logarithm* of $n/k$, not on $n$ itself, so a stream a thousand times longer adds only a few entries per slot.
- $k/i$ is the chance that item $i$ ranks in the top $k$ of the first $i$ items. **Matters:** it falls as the stream goes on: late items rarely beat the current best $k$.
- $H_m=1+\frac12+\dots+\frac1m$ is the $m$-th harmonic number. **Matters:** the sum $\sum_{i=k+1}^n 1/i$ equals $H_n-H_k$, and $H_m$ grows only like $\ln m$ (slowly).
- "Random order" is an assumption. **Matters:** if the stream arrives sorted from small to large, *every* item enters and the cost is $n$ entries. The formula covers only a randomly shuffled stream.

**Example (tiny numbers):** $n=4$, $k=2$.
1. $H_4=1+\frac12+\frac13+\frac14=2.0833$ and $H_2=1.5$. So $\mathbb E=2\,(1+2.0833-1.5)=2\times1.5833=3.1667$.
2. The code averaged the entries over all 24 orders of four numbers and got 3.1667, which is the same.
3. For $n=100{,}000$ and $k=10$ the formula gives 101.6 and the code's average over 300 shuffles was 101.0 (a single run gave 104).
4. Meaning: finding the 10 biggest of 100,000 numbers needs only about 100 heap operations, each $O(\log k)$, plus 100,000 quick comparisons with the smallest of the ten. Sorting everything would be a much larger job.

**Other points (from memory, not checked here):**
- Python's `heapq` is documented as a binary heap held in a list. I did not read its source, and I did not check how its `heapify` and `nlargest` are implemented.
- Dijkstra's shortest-path algorithm uses a priority queue, and its running time depends on the heap used. It has its own note, [[Dijkstra's Algorithm]], where I ran it with `heapq`. How the running time changes with other heaps is from memory.
- Other priority-queue structures (balanced trees, Fibonacci heaps, pairing heaps) exist with different trade-offs. I did not cover them.
- Heapsort is not *stable* (equal items can change their order), and it is usually less cache-friendly than quicksort. Both claims are from memory and untested here.
- A heap is not good at searching for a given item (it is $O(n)$), or at listing the items in order without removing them.

**What came before, and how this improved on it:**
- **Before:** keep items in a list and scan the whole list to find the smallest each time.
- **Its limit:** each pick costs about $n$ comparisons.
- **What changed:** a heap keeps the smallest item at the top and repairs itself in about $\log_2 n$ steps. J. W. J. Williams described it in 1964 for heapsort (from memory, not checked).
- **How much better:** in the code below, taking the smallest of 5,000 items 5,000 times needed 12,497,500 comparisons by scanning and 63,324 with a heap, 197 times fewer, and the order matched a full sort.

**Check it yourself:**

```python
import heapq, random
random.seed(0)
class C:
    n = 0
    def __init__(s, v): s.v = v
    def __lt__(s, o): C.n += 1; return s.v < o.v
n = 5000
data = [random.random() for _ in range(n)]
scan = 0; pool = data[:]
for _ in range(n):                                       # take the smallest by scanning every time
    m = 0
    for i in range(1, len(pool)):
        scan += 1
        if pool[i] < pool[m]: m = i
    pool.pop(m)
h = [C(v) for v in data]; C.n = 0; heapq.heapify(h)
out = [heapq.heappop(h).v for _ in range(n)]
print(f"taking the smallest item {n:,} times: scanning {scan:,} comparisons, heap {C.n:,} ({scan // C.n}x fewer); same order: {out == sorted(data)}")
```

## Related
[[Arrays and Lists]] · [[Stacks and Queues]] · [[Sorting]] · [[Big-O]] · [[Binary Search]] · [[Graphs]] · [[Recursion]] · [[Hash Tables]] · [[Vector Database]] · [[Python]] · [[Trees]] · [[Dijkstra's Algorithm]] · [[Greedy Algorithms]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, chapter on heapsort and priority queues (from memory; I did not open it for this note).
- Python documentation for the `heapq` module (docs.python.org).
- Figures: [Binary_heap_indexing.png](https://commons.wikimedia.org/wiki/File:Binary_heap_indexing.png) (Nagae, CC BY 2.5) and [InsertingIntoBinaryHeap.gif](https://commons.wikimedia.org/wiki/File:InsertingIntoBinaryHeap.gif) (Sanglee123, CC0), on Wikimedia Commons. I viewed the first and all 5 frames of the second before writing the captions.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/data-structures Heap (priority queue):::A hospital waiting room where the most urgent patient is seen first, whatever time they arrived. A heap keeps this organised so finding and removing the best is quick.

#flashcards/algorithms/concept #flashcards/topic/data-structures What is the heap rule?::Picture a pyramid with the best item at the top, filling each row from the left. Every item is better than or equal to the two items directly below it, so the best of all is always at the top.

#flashcards/algorithms/concept #flashcards/topic/data-structures How is the pyramid stored?::In a plain list, read row by row, left to right. No pointers are needed: a tiny rule gives each item's children and parent.

#flashcards/algorithms/concept #flashcards/topic/data-structures How do you add an item, and how do you take out the best?::Add: put it at the end of the list and swap it up with its parent while it is better (sifting up, at most the height of the pyramid). Take out: remove the top, move the last item to the top, and swap it down with the better child while it is worse (sifting down).

#flashcards/algorithms/concept #flashcards/topic/data-structures Why is a heap fast?::A pyramid with a million items is only about 20 rows high, so each push or pop takes at most about 20 swaps.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the positions of a heap item's children and parent and explain each symbol.
?
$$\text{left}(i)=2i+1,\qquad \text{right}(i)=2i+2,\qquad \text{parent}(i)=\left\lfloor\frac{i-1}{2}\right\rfloor$$
- $i$: a position in the list (0 is the top). These rules are all a heap needs to act like a tree, with no pointers and no extra memory.
- $2i+1$, $2i+2$: the left and right child. Row $r$ holds positions $2^r-1$ to $2^{r+1}-2$, so doubling $i$ lands on the next row.
- $\lfloor\cdot\rfloor$: round down. Both children map back to $i$: $\lfloor 2i/2\rfloor=i$ and $\lfloor(2i+1)/2\rfloor=i$.
- If you count from 1 instead of 0 the rules become $2i$, $2i+1$ and $\lfloor i/2\rfloor$.

#flashcards/algorithms/example #flashcards/topic/data-structures The heap list is `[15, 5, 11, 3, 4, 8]`. What are the parent of position 5 and the children of position 1?::Position 5 holds 8, its parent is $\lfloor(5-1)/2\rfloor=2$, holding 11 (8 is the child of 11 in the picture). Position 1 holds 5, its children are $3$ and $4$, holding 3 and 4. Every parent is at least as big as its children.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the bound on the swaps needed to build a heap and explain each symbol.
?
$$\text{swaps}\ \le\ \sum_{v}h(v)\ =\ n-\operatorname{popcount}(n)\ <\ n$$
- $h(v)$: the height of position $v$, how many steps down the longest path from it goes. Half of the items are leaves (height 0) and need no work, a quarter have height 1, an eighth height 2, and so on. The cheap items are the many, the costly ones the few.
- $\sum_v h(v)$: the worst case for every position, the most a build can ever swap whatever the input order.
- $\operatorname{popcount}(n)$: the number of 1s when $n$ is written in binary. The sum of heights equals $n-\operatorname{popcount}(n)$.

#flashcards/algorithms/example #flashcards/topic/data-structures $n=7$. What is the sum of heights, and what about $n=1000$?::Heights: top 2, two middle items 1 each, four leaves 0, sum 4. $7-\operatorname{popcount}(7)=7-3=4$. The code's swaps for 7 items were 4 for the worst order tried. For $n=1000$ the bound is 994 and the falling-order input needed 992. Building a heap takes at most about one swap per item.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the expected heap entries when keeping the top $k$ of a stream and explain each symbol.
?
$$\mathbb E[\text{entries}]=k+\sum_{i=k+1}^{n}\frac{k}{i}=k\,\big(1+H_n-H_k\big)\ \approx\ k\,\Big(1+\ln\frac nk\Big)$$
- $n$: the number of items in the stream. $k$: how many you keep. The number of entries depends on $k$ and the logarithm of $n/k$, not on $n$ itself.
- $k/i$: the chance that item $i$ ranks in the top $k$ of the first $i$ items. It falls as the stream goes on, because late items rarely beat the current best $k$.
- $H_m=1+\frac12+\dots+\frac1m$: the $m$-th harmonic number. $\sum_{i=k+1}^n 1/i=H_n-H_k$, and $H_m$ grows like $\ln m$.

#flashcards/algorithms/example #flashcards/topic/data-structures $n=4$, $k=2$ and then $n=100{,}000$, $k=10$. What is the expected number of entries?::$H_4=2.0833$, $H_2=1.5$, so $2(1+2.0833-1.5)=3.1667$ (the code averaged all 24 orders and got 3.1667). For $n=100{,}000$, $k=10$ the formula gives 101.6 and the code's average over 300 shuffles was 101.0. The 10 biggest of 100,000 numbers need only about 100 heap operations.

#flashcards/algorithms/image #flashcards/topic/data-structures ![[binary-heap-indexing.png]]
What does this picture show?
?
The shape of a binary heap with 12 items, numbered row by row from 1 (items 2 and 3 are the children of item 1, items 4 and 5 the children of item 2, and so on). In Python the list starts at 0, so the numbers in the code are one less. (From the note's caption: Nagae, Wikimedia Commons.)

#flashcards/algorithms/image #flashcards/topic/data-structures ![[heap-insert-animation.gif]]
What does this picture show?
?
Adding 15 to a max-heap (the biggest value is kept at the top). The new value goes in the next free spot at the bottom and "bubbles up" past 8 and then past 11 until it reaches the top. It is an animation, and some viewers show only the first frame. (From the note's caption: Sanglee123, Wikimedia Commons, CC0.)
