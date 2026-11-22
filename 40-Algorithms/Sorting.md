---
tags: [algorithms, sorting, merge-sort, quicksort, programming, python]
status: growing
level: 4
tier: 2
---

# Sorting

Sorting means **putting items in order**, such as numbers from smallest to largest or names from A to Z. It sounds simple, but it is one of the most studied problems in computing, because so many other things (searching, finding duplicates, ranking) become faster or easier once the data is sorted. The simplest methods take about $n^2$ steps. The good ones take about $n\log n$, and it is proved that no method based on comparisons can do much better (see [[Big-O]]).

## 🟢 1. Simple (age 12)
You have a hand of playing cards in a mess and want them in order. One way: pick up cards one at a time and slide each into its right place among the ones you already hold. That works, but with 1,000 cards it is slow, because each new card may have to be compared with many others. A faster way: split the pile into two halves, sort each half, and then **merge** the two sorted halves by repeatedly taking the smaller of the two top cards. Because each half is already sorted, merging is quick.

**Example:** A teacher has 8 tests to put in order of marks. Instead of searching the whole pile each time, she splits it into two piles of 4, then each into two of 2. Two papers are easy to put in order. Then she combines the little sorted piles into bigger sorted ones, 2 and 2 into 4, 4 and 4 into 8. At every step she only has to look at the top paper of each pile.

## 🔵 2. How it works
Two classic fast methods, in words.

**Merge sort (split, then merge):**
1. If the list has 0 or 1 items, it is already sorted. Stop.
2. **Split** the list into two halves.
3. **Sort each half** (by doing the same thing again on it).
4. **Merge** the two sorted halves: look at the first item of each, take the smaller one, and repeat until both are used up.

**Quicksort (pick a pivot, then split around it):**
1. If the list has 0 or 1 items, it is already sorted. Stop.
2. **Pick one item**, called the *pivot*.
3. **Partition:** put everything smaller than the pivot on its left, and everything else on its right. The pivot is now in its final place.
4. **Sort the left part and the right part** (by doing the same thing again on each).

![[merge-sort-diagram.svg]]
*Merge sort on the list 38, 27, 43, 3, 9, 82, 10. Red arrows going down: the list is split in two again and again until single items remain. Green arrows going up: the small sorted pieces are merged back into bigger sorted ones, ending with 3, 9, 10, 27, 38, 43, 82 at the bottom. Image: VineetKumar (adapted from an earlier public-domain diagram), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Merge_sort_algorithm_diagram.svg), public domain.*

## 🟠 3. Technical detail
- **What counts as work:** sorting methods are compared by the number of **comparisons** ("is this item smaller than that one?") and the number of moves. The code counts comparisons, because the count is the same on every computer.
- **The simple methods are $O(n^2)$:**
  - **Insertion sort** (the cards in your hand): about $n^2/4$ comparisons on random data, up to $n(n-1)/2$ in the worst case, but only about $n$ if the list is already nearly sorted.
  - **Selection sort** (repeatedly pick the smallest of what is left) always makes $n(n-1)/2$ comparisons. **Bubble sort** is similar and is mainly of teaching interest.
  - They are fine for small lists, and simple to write.
- **Merge sort is $O(n\log n)$ always:** it splits into halves about $\log_2 n$ times, and each level of merging reads every item once. It needs extra memory for merging (about $n$ items). It is **stable** (equal items keep their original order, see below).
- **Quicksort is $O(n\log n)$ on average and $O(n^2)$ in the worst case:** if the pivot is always the smallest or largest item (for example the last item of an already sorted list), one side is empty each time and the work is quadratic (the code shows 124,750 comparisons for 500 sorted items). Choosing the pivot *at random* makes this very unlikely. It sorts in place (little extra memory) and is often fast in practice, because it works well with the computer's memory cache (from memory).
- **The comparison lower bound:** any method that only compares items needs at least $\lceil\log_2(n!)\rceil$ comparisons in the worst case, which is $\Omega(n\log n)$ (checked online; see [[Big-O]]). So $O(n\log n)$ is the best possible for such methods, and merge sort is close to it (the code compares its worst case with the bound for $n\le8$).
- **Stable sorting:** a sort is *stable* if items that compare equal stay in their original order. This matters when you sort by one column and then another, for example by city and then by name. Python's `sorted` and `list.sort` are stable (the code shows it).
- **Python's sort** is a hybrid called Timsort (from memory), which takes advantage of parts of the data that are already in order. The code measures this: for 10,000 numbers, Python's `sorted` made 119,835 comparisons on random data (the minimum possible is 118,459) but only 9,999 on data that was already sorted or reverse sorted. In practice, use `sorted()` and `sort(key=...)` and do not write your own.
- **Not based on comparisons:** if the keys are small whole numbers, methods such as counting sort and radix sort can sort in about $O(n)$ time, because they do not compare items. They are outside the lower bound (from memory; not covered in the code).
- **External and parallel sorting:** data too large for memory is sorted in pieces that are merged (the merge step again), and sorting is run in parallel on many processors or on a [[GPU]]. Neither is covered here.
- **Where sorting shows up in this vault:**
  - [[Binary Search]] needs sorted data.
  - [[Pandas]] sorts a table with `sort_values`.
  - The top-$k$ results in a [[Vector Database]] search or in a language model's [[Softmax]] output need the largest items, and for that you often do *not* need a full sort (from memory: a partial sort or a heap is enough).
  - A [[Decision Tree]] considers sorted values of a feature when it looks for the best split.
  - The ranks used in some [[Evaluation Metrics]] come from sorting scores.
- **Limits:** the code counts comparisons on small random lists. It does not time the programs, and it does not test large data, strings, or lists with many equal items.

![[quicksort-diagram.svg]]
*Quicksort on the list 3, 7, 8, 5, 2, 1, 9, 5, 4. The last item (4, shown in black) is the pivot. After one partition, smaller items (3, 1, 2) are on its left and larger ones (5, 8, 9, 5, 7) on its right, and 4 is in its final place. The same is then done to each side, and the sorted list 1, 2, 3, 4, 5, 5, 7, 8, 9 comes out at the bottom. Image: Znupi, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Quicksort-diagram.svg), public domain.*

The code checks six things, all by counting comparisons: (1) the two picture lists; (2) random lists of three sizes against the formulas; (3) the exact worst case of merge sort for every ordering of up to 8 items; (4) quicksort on sorted data and with a random pivot; (5) stability; and (6) the number of comparisons that Python's own `sorted` makes.

```python
import itertools, math, random
def insertion(a):                                           # returns (sorted copy, comparisons)
    a = a[:]; c = 0
    for i in range(1, len(a)):
        j = i
        while j > 0:
            c += 1
            if a[j - 1] > a[j]: a[j - 1], a[j] = a[j], a[j - 1]; j -= 1
            else: break
    return a, c
def merge_sort(a):
    if len(a) <= 1: return a[:], 0
    mid = (len(a) + 1) // 2                                  # left half gets the extra item, as in the picture
    L, cl = merge_sort(a[:mid]); R, cr = merge_sort(a[mid:]); out, i, j, c = [], 0, 0, 0
    while i < len(L) and j < len(R):
        c += 1
        if L[i] <= R[j]: out.append(L[i]); i += 1            # <= keeps equal items in their original order (stable)
        else: out.append(R[j]); j += 1
    return out + L[i:] + R[j:], cl + cr + c
def quick(a, pivot='last', rng=None):
    if len(a) <= 1: return a[:], 0
    p = a[-1] if pivot == 'last' else a[rng.randrange(len(a))]
    rest = list(a); rest.remove(p)
    lo = [x for x in rest if x < p]; hi = [x for x in rest if x >= p]
    sl, cl = quick(lo, pivot, rng); sh, ch = quick(hi, pivot, rng)
    return sl + [p] + sh, cl + ch + len(rest)                # every other item is compared once with the pivot
d = [38, 27, 43, 3, 9, 82, 10]
print('picture list:', merge_sort(d)[0], '| merge comparisons', merge_sort(d)[1], '| insertion', insertion(d)[1], '| quick (last pivot)', quick(d)[1])
d2 = [3, 7, 8, 5, 2, 1, 9, 5, 4]; print('quicksort picture list:', quick(d2)[0], '| pivot 4 splits into', [x for x in d2 if x < 4], 'and', [x for x in d2[:-1] if x >= 4])
# 1) comparisons on random lists: counted, not timed
rng = random.Random(0)
for n in (100, 1000, 2000):
    a = [rng.random() for _ in range(n)]
    print(n, '| insertion', insertion(a)[1], '(~n^2/4 =', n * n // 4, ') | merge', merge_sort(a)[1], '| quick', quick(a)[1], '| n*log2(n) =', round(n * math.log2(n)), '| minimum possible ceil(log2 n!) =', math.ceil(math.lgamma(n + 1) / math.log(2)))
# 2) exact worst case for merge sort, by trying every ordering of n items
for n in range(2, 9):
    worst = max(merge_sort(list(p))[1] for p in itertools.permutations(range(n)))
    k = math.ceil(math.log2(n)); print('n', n, '| worst merge comparisons', worst, '| formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 =', n * k - 2 ** k + 1, '| lower bound ceil(log2 n!) =', math.ceil(math.log2(math.factorial(n))))
# 3) quicksort: a bad pivot rule on sorted input, and the average with a random pivot
n = 500; s = list(range(n)); print('sorted input, last-item pivot:', quick(s)[1], 'comparisons | n(n-1)/2 =', n * (n - 1) // 2)
H = lambda m: sum(1 / i for i in range(1, m + 1))
for n in (3, 1000):
    avg = sum(quick([rng.random() for _ in range(n)], 'random', rng)[1] for _ in range(300 if n == 1000 else 20000)) / (300 if n == 1000 else 20000)
    print('random pivot, n', n, '| simulated average', round(avg, 2), '| formula 2(n+1)H_n - 4n =', round(2 * (n + 1) * H(n) - 4 * n, 2))
# 4) stable sorting, and Python's own sort counting its comparisons
recs = [('b', 2), ('a', 1), ('c', 2), ('d', 1), ('e', 2)]
print('sorted by number only, ties keep their order:', [r[0] for r in sorted(recs, key=lambda r: r[1])])
class C:
    n = 0
    def __init__(self, v): self.v = v
    def __lt__(self, o): C.n += 1; return self.v < o.v
m = 10000; vals = [rng.random() for _ in range(m)]
for name, data in (('random', vals), ('already sorted', sorted(vals)), ('sorted backwards', sorted(vals, reverse=True))):
    C.n = 0; sorted([C(v) for v in data]); print('Python sorted(), 10000', name, '->', C.n, 'comparisons')
print('n*log2(n) for 10000 =', round(m * math.log2(m)), '| minimum possible ceil(log2 n!) =', math.ceil(math.lgamma(m + 1) / math.log(2)))
# picture list: [3, 9, 10, 27, 38, 43, 82] | merge comparisons 14 | insertion 15 | quick (last pivot) 13
# quicksort picture list: [1, 2, 3, 4, 5, 5, 7, 8, 9] | pivot 4 splits into [3, 2, 1] and [7, 8, 5, 9, 5]
# 100 | insertion 2436 (~n^2/4 = 2500 ) | merge 533 | quick 588 | n*log2(n) = 664 | minimum possible ceil(log2 n!) = 525
# 1000 | insertion 244993 (~n^2/4 = 250000 ) | merge 8753 | quick 11645 | n*log2(n) = 9966 | minimum possible ceil(log2 n!) = 8530
# 2000 | insertion 1003986 (~n^2/4 = 1000000 ) | merge 19440 | quick 25419 | n*log2(n) = 21932 | minimum possible ceil(log2 n!) = 19053
# n 2 | worst merge comparisons 1 | formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 = 1 | lower bound ceil(log2 n!) = 1
# n 3 | worst merge comparisons 3 | formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 = 3 | lower bound ceil(log2 n!) = 3
# n 4 | worst merge comparisons 5 | formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 = 5 | lower bound ceil(log2 n!) = 5
# n 5 | worst merge comparisons 8 | formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 = 8 | lower bound ceil(log2 n!) = 7
# n 6 | worst merge comparisons 11 | formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 = 11 | lower bound ceil(log2 n!) = 10
# n 7 | worst merge comparisons 14 | formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 = 14 | lower bound ceil(log2 n!) = 13
# n 8 | worst merge comparisons 17 | formula n*ceil(log2 n) - 2^ceil(log2 n) + 1 = 17 | lower bound ceil(log2 n!) = 16
# sorted input, last-item pivot: 124750 comparisons | n(n-1)/2 = 124750
# random pivot, n 3 | simulated average 2.67 | formula 2(n+1)H_n - 4n = 2.67
# random pivot, n 1000 | simulated average 11014.69 | formula 2(n+1)H_n - 4n = 10985.91
# sorted by number only, ties keep their order: ['a', 'd', 'b', 'c', 'e']
# Python sorted(), 10000 random -> 119835 comparisons
# Python sorted(), 10000 already sorted -> 9999 comparisons
# Python sorted(), 10000 sorted backwards -> 9999 comparisons
# n*log2(n) for 10000 = 132877 | minimum possible ceil(log2 n!) = 118459
```
Reading the output (Python 3.9, standard library only; fixed seeds):
- **The picture lists:** merge sort sorts the 7-number list with 14 comparisons, insertion sort with 15 and quicksort (last item as pivot) with 13. In the quicksort picture list, pivot 4 splits the rest into $[3,2,1]$ and $[7,8,5,9,5]$, the same two groups as in the figure (the order inside each group differs, because the picture partitions in place and my code builds new lists).
- **Three sizes:** for 1,000 random numbers, insertion sort needs 244,993 comparisons (close to $n^2/4=250{,}000$), merge sort 8,753 and quicksort 11,645. The minimum possible for any comparison sort is at least 8,530. So merge sort is within about 3% of the best possible here. At 2,000 numbers the insertion count rises four times (to 1,003,986), while merge sort's rises a little more than two times (to 19,440).
- **Worst case of merge sort:** trying all orderings, the worst case for $n=2,\dots,8$ matches the formula $n\lceil\log_2n\rceil-2^{\lceil\log_2n\rceil}+1$ exactly (for example 17 for $n=8$). It is at or just above the lower bound (17 against 16 for $n=8$, and equal for $n\le4$).
- **Quicksort's bad day:** on 500 already-sorted numbers with the last item as pivot, it made 124,750 comparisons, exactly $n(n-1)/2$.
- **Random pivot:** the average for $n=3$ is 2.67, matching $8/3$ from the formula. For $n=1000$ the simulated average (300 runs) was 11,014.69 against the formula's 10,985.91. They differ by 0.3%, which is within the noise of 300 runs.
- **Stability:** sorting by the number only, items with equal numbers kept their original order: a before d, and b, c, e in their original order.
- **Python's sort:** 10,000 random numbers took 119,835 comparisons (the minimum possible is 118,459, and $n\log_2n$ is 132,877). Data already sorted, or sorted backwards, took only 9,999, just $n-1$. So Python's sort is adaptive: it notices ordered runs.
- **Limits:** these are counts on small random lists in simple code. They say nothing about speed, large data or other kinds of input.

## 🔴 4. Deep / formal

**Merge sort's worst case.** Merging two sorted lists of sizes $a$ and $b$ takes at most $a+b-1$ comparisons. Splitting $n$ items into halves, sorting each half and merging gives, in the worst case,

$$C(n)=n\lceil\log_2n\rceil-2^{\lceil\log_2n\rceil}+1\ \ \le\ n\log_2n\ \text{comparisons}$$

**Where:**
- $n$ is the number of items. **Matters:** the cost grows a little faster than $n$ (the $\log n$ factor comes from the number of levels of splitting).
- $\lceil\log_2n\rceil$ is the number of times you can halve $n$ before reaching single items, rounded up. **Matters:** it is the number of levels in the split-and-merge tree (the picture has 3 levels for $n=7$).
- $2^{\lceil\log_2n\rceil}$ is the next power of two at or above $n$. **Matters:** when $n$ is exactly a power of two, the formula gives $n\log_2n-n+1$. Between powers of two, the cost is a bit lower than $n\log_2n$ would suggest.
- The $+1$ and the subtracted power of two come from merging being $a+b-1$ (not $a+b$) and from the tree being uneven when $n$ is not a power of two. **Matters:** this makes the formula exact, which the code checked for $n=2$ to $8$ by trying every ordering. I did not prove it.
- Dropping the lower terms gives $O(n\log n)$. **Matters:** this is the shape that appears in [[Big-O]].

**Example (tiny numbers):**
1. $n=5$: $\lceil\log_25\rceil=3$ and $2^3=8$. $C(5)=5\cdot3-8+1=8$.
2. $n=8$: $\lceil\log_28\rceil=3$. $C(8)=8\cdot3-8+1=17$.
3. The code tried all $5!=120$ and all $8!=40{,}320$ orderings and found the worst cases 8 and 17.
4. Compare the lower bound $\lceil\log_2(n!)\rceil$: 7 for $n=5$ and 16 for $n=8$. Merge sort is one comparison above the bound in both cases.
5. Meaning: for 8 items, merge sort never needs more than 17 comparisons, while the theoretical minimum for any comparison method is 16. Insertion sort can need up to $8\cdot7/2=28$.

**The simple methods.** Selection sort compares each item with all those after it, and insertion sort in its worst case does too, so both need

$$\binom n2=\frac{n(n-1)}{2}=O(n^2)\ \text{comparisons (worst case)}$$

**Where:**
- $n$ is the number of items. **Matters:** doubling $n$ makes the work about four times larger (the code shows 244,993 going to 1,003,986).
- $\binom n2$ is the number of pairs of items, as in the handshake count in [[Big-O]]. **Matters:** the worst case compares every pair once.
- On random data, insertion sort does about half as many, $n^2/4$. **Matters:** an average item travels about halfway back through the sorted part. The code measured 244,993 for $n=1000$ against $n^2/4=250{,}000$.
- On data that is already sorted, insertion sort needs only $n-1$ comparisons. **Matters:** the cost depends on how disordered the data is, which is why it is useful for nearly sorted data and why Python's sort uses ideas from it (from memory).

**Example (tiny numbers):**
1. $n=5$: $5\cdot4/2=10$ comparisons at most.
2. $n=1000$: $1000\cdot999/2=499{,}500$ at the worst, about $250{,}000$ on random data.
3. Merge sort for 1,000 items needs 8,753 on random data (measured), so about 28 times fewer than the 244,993 for insertion sort.
4. Meaning: for 1,000 items the $n\log n$ method does under 4% of the work. For 1,000,000 items the quadratic methods would need on the order of $10^{12}$ comparisons, which is unusable.

**Quicksort's average cost.** With a pivot chosen uniformly at random, the expected number of comparisons to sort $n$ distinct items is

$$E[C_n]=2(n+1)H_n-4n\ \approx\ 1.39\,n\log_2n,\qquad H_n=1+\tfrac12+\tfrac13+\dots+\tfrac1n,\qquad \text{worst case } \tfrac{n(n-1)}{2}$$

**Where:**
- $H_n$ is the $n$-th harmonic number, the sum of $1/i$ from 1 to $n$. **Matters:** it grows like $\ln n$, which is where the $n\log n$ comes from.
- $2(n+1)H_n$ is the main term. **Matters:** a random pivot splits the list in a balanced enough way on average, and the factor 2 reflects that each pivot is compared with all the others in its part.
- $-4n$ is a lower-order correction. **Matters:** it makes the formula exact for small $n$, as the example shows.
- The approximation $1.39\,n\log_2n$ is a rounded version of the leading term, $2n\ln n=1.386\,n\log_2 n$. **Matters:** it shows quicksort uses about 39% more comparisons than the minimum $n\log_2n$ on average. Its speed in practice comes partly from other factors such as memory use (from memory).
- The worst case $n(n-1)/2$ happens if every pivot is the smallest or largest item. **Matters:** a random pivot makes this extremely unlikely, which is the reason to choose it at random. I did not prove the formula. I checked it with $n=3$ by hand and by simulation.

**Example (tiny numbers):**
1. $n=3$: $H_3=1+0.5+0.333=1.8333$. $E[C_3]=2\cdot4\cdot1.8333-12=14.667-12=2.667=8/3$.
2. Check by hand: the pivot is the middle item with chance $1/3$, which costs 2 comparisons and leaves two single items. Otherwise (chance $2/3$) it costs 2 comparisons plus 1 more for the leftover pair, which is 3. Average: $\tfrac13\cdot2+\tfrac23\cdot3=\tfrac83=2.667$.
3. The code's simulation for $n=3$ gave 2.67.
4. $n=1000$: $H_{1000}=7.4855$. $E[C]=2\cdot1001\cdot7.4855-4000=10{,}985.9$. The simulation gave 11,014.7.
5. Meaning: about 11,000 comparisons for 1,000 items, close to the 8,530 minimum (about 29% higher), and far below the worst case of 499,500.

**Other points (from memory, not checked here):**
- Merge sort is credited to John von Neumann (1945) and quicksort to Tony Hoare (around 1960). I did not check either attribution.
- Timsort was designed for Python by Tim Peters. I did not check its details, and my statement that Python's sort uses ideas of insertion sort and merging is from memory. The comparison counts above are measured.
- The $n\log_2 n$ minimum is a bound on *comparisons* only. Methods that use the values themselves (counting sort, radix sort) escape it for restricted kinds of keys.
- Heap sort, which has $O(n\log n)$ worst case and sorts in place, is not covered here.

**What came before, and how this improved on it:**
- **Before:** simple sorts such as bubble sort, which compare neighbours again and again, about $n^2/2$ comparisons.
- **Its limit:** fine for 100 items, hopeless for a million.
- **What changed:** divide and conquer, as in merge sort: split in halves, sort each, merge. John von Neumann described merge sort in 1945 (from memory, not checked). It needs about $n\log_2 n$ comparisons.
- **How much better:** in the code below, for 2,000 random numbers bubble sort used 1,999,000 comparisons and merge sort used 19,484, which is 103 times fewer. For 500 numbers the gap was 32 times, so it widens as $n$ grows. Both gave the same sorted list.

**Check it yourself:**

```python
import random
def bubble(a):
    a, c = a[:], 0
    for i in range(len(a)):
        for j in range(len(a) - 1 - i):
            c += 1
            if a[j] > a[j + 1]: a[j], a[j + 1] = a[j + 1], a[j]
    return a, c
def merge_sort(a):
    if len(a) <= 1: return a, 0
    l, cl = merge_sort(a[:len(a) // 2]); r, cr = merge_sort(a[len(a) // 2:])
    out, i, j, c = [], 0, 0, cl + cr
    while i < len(l) and j < len(r):
        c += 1
        if l[i] <= r[j]: out.append(l[i]); i += 1
        else: out.append(r[j]); j += 1
    return out + l[i:] + r[j:], c
random.seed(0)
for n in (500, 2000):
    a = random.sample(range(10 * n), n)
    (sb, cb), (sm, cm) = bubble(a), merge_sort(a)
    print(f"n={n}: bubble sort {cb:,} comparisons, merge sort {cm:,}  ({cb / cm:.0f}x fewer)   both sorted correctly: {sb == sm == sorted(a)}")
```

## Related
[[Big-O]] · [[Binary Search]] · [[Hash Tables]] · [[Recursion]] · [[Python]] · [[Pandas]] · [[NumPy]] · [[GPU]] · [[Decision Tree]] · [[Vector Database]] · [[Softmax]] · [[Evaluation Metrics]] · [[Arrays and Lists]] · [[Heaps and Priority Queues]] · [[Trees]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, the chapters on merge sort and quicksort (from memory; I did not open it for this note).
- Knuth, *The Art of Computer Programming*, volume 3, *Sorting and Searching* (from memory; I did not open it).
- Wikipedia, [Comparison sort](https://en.wikipedia.org/wiki/Comparison_sort): the $\lceil\log_2(n!)\rceil$ lower bound (checked online).
- Python documentation: the *Sorting HOW TO*, on `sorted`, `key` and stability (docs.python.org).
- Figures: [Merge_sort_algorithm_diagram.svg](https://commons.wikimedia.org/wiki/File:Merge_sort_algorithm_diagram.svg) (VineetKumar, public domain) and [Quicksort-diagram.svg](https://commons.wikimedia.org/wiki/File:Quicksort-diagram.svg) (Znupi, public domain), both on Wikimedia Commons. I viewed both before writing the captions.
- The code, the brute-force checks and every Example number were run for this note (Python 3.9, standard library only). Comparisons were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/searching-sorting Merge sort:::Split the list in two halves, sort each half (by doing the same again), then merge the two sorted halves: look at the first item of each, take the smaller, and repeat.

#flashcards/algorithms/reverse #flashcards/topic/searching-sorting Quicksort:::Pick a pivot, put everything smaller on its left and everything else on its right (the pivot is now in its final place), then sort each part by doing the same again.

#flashcards/algorithms/formula #flashcards/topic/searching-sorting Write the comparisons of merge sort and explain each symbol.
?
$$C(n)=n\lceil\log_2n\rceil-2^{\lceil\log_2n\rceil}+1\ \ \le\ n\log_2n\ \text{comparisons}$$
- $n$: the number of items. The cost grows a little faster than $n$, the $\log n$ factor coming from the levels of splitting.
- $\lceil\log_2n\rceil$: the number of times you can halve $n$ before reaching single items, rounded up. It is the number of levels in the split-and-merge tree (3 levels for $n=7$ in the picture).
- $2^{\lceil\log_2n\rceil}$: the next power of two at or above $n$. When $n$ is exactly a power of two the formula gives $n\log_2n-n+1$.

#flashcards/algorithms/example #flashcards/topic/searching-sorting What are $C(5)$ and $C(8)$, and how do they compare with the lower bound?::$C(5)=5\cdot3-8+1=8$. $C(8)=8\cdot3-8+1=17$. The code tried all $5!=120$ and $8!=40{,}320$ orderings and found worst cases 8 and 17. The lower bound $\lceil\log_2(n!)\rceil$ is 7 for $n=5$ and 16 for $n=8$, so merge sort is one comparison above it in both. For 8 items no comparison method can do better than 16.

#flashcards/algorithms/formula #flashcards/topic/searching-sorting Write the comparisons of insertion sort in the worst case and explain each symbol.
?
$$\binom n2=\frac{n(n-1)}{2}=O(n^2)\ \text{comparisons (worst case)}$$
- $n$: the number of items. Doubling $n$ makes the work about four times larger (the code shows 244,993 going to 1,003,986).
- $\binom n2$: the number of pairs of items, as in the handshake count. The worst case compares every pair once.
- On random data insertion sort does about half as many, $n^2/4$, because an average item travels about halfway back through the sorted part. The code measured 244,993 for $n=1000$ against $n^2/4=250{,}000$.

#flashcards/algorithms/example #flashcards/topic/searching-sorting How many comparisons for $n=5$ and $n=1000$, and how does merge sort compare?::$n=5$: at most 10. $n=1000$: at most 499,500, about 250,000 on random data. Merge sort needs 8,753 on random data (measured), about 28 times fewer than insertion sort's 244,993, under 4% of the work. For a million items the quadratic methods would need about $10^{12}$ comparisons.

#flashcards/algorithms/formula #flashcards/topic/searching-sorting Write the expected comparisons of quicksort and explain each symbol.
?
$$E[C_n]=2(n+1)H_n-4n\ \approx\ 1.39\,n\log_2n,\qquad H_n=1+\tfrac12+\tfrac13+\dots+\tfrac1n,\qquad \text{worst case } \tfrac{n(n-1)}{2}$$
- $H_n$: the $n$-th harmonic number, the sum of $1/i$ from 1 to $n$. It grows like $\ln n$, which is where $n\log n$ comes from.
- $2(n+1)H_n$: the main term. A random pivot splits the list in a balanced enough way on average, and the factor 2 reflects that each pivot is compared with all the others in its part.
- $-4n$: a lower-order correction that makes the formula exact for small $n$.
- $1.39\,n\log_2n$: a rounded version of the leading term $2n\ln n$.
- The worst case is $n(n-1)/2$.

#flashcards/algorithms/example #flashcards/topic/searching-sorting What is $E[C_3]$, and $E[C_{1000}]$?::$H_3=1.8333$, $E[C_3]=2\cdot4\cdot1.8333-12=2.667=8/3$. By hand: the pivot is the middle item with chance $\tfrac13$ (2 comparisons), otherwise 3: $\tfrac13\cdot2+\tfrac23\cdot3=\tfrac83$. The simulation gave 2.67. For $n=1000$: $H_{1000}=7.4855$, $E[C]=10{,}985.9$, and the simulation gave 11,014.7.

#flashcards/algorithms/image #flashcards/topic/searching-sorting ![[merge-sort-diagram.svg]]
What does this picture show?
?
Merge sort on the list 38, 27, 43, 3, 9, 82, 10. Red arrows going down: the list is split in two again and again until single items remain. Green arrows going up: the small sorted pieces are merged back into bigger sorted ones, ending with 3, 9, 10, 27, 38, 43, 82. (From the note's caption: VineetKumar, Wikimedia Commons, public domain.)

#flashcards/algorithms/image #flashcards/topic/searching-sorting ![[quicksort-diagram.svg]]
What does this picture show?
?
Quicksort on the list 3, 7, 8, 5, 2, 1, 9, 5, 4. The last item (4, in black) is the pivot. After one partition smaller items (3, 1, 2) are on its left and larger ones (5, 8, 9, 5, 7) on its right, and 4 is in its final place. The same is then done to each side and the sorted list 1, 2, 3, 4, 5, 5, 7, 8, 9 comes out at the bottom. (From the note's caption: Znupi, Wikimedia Commons, public domain.)
