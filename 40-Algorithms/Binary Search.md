---
tags: [algorithms, searching, binary-search, bisection, programming, python]
status: growing
level: 4
tier: 1
---

# Binary Search

Binary search finds an item in a **sorted** list by always looking at the middle, and throwing away the half that cannot contain it. Each look halves the work, so even a list of a million items needs only about 20 looks. It is the standard example of $O(\log n)$ in [[Big-O]], and the same halving idea finds answers to problems that are not lists at all (for example, a root of an equation).

## 🟢 1. Simple (age 12)
You think of a number from 1 to 100, and I must guess it. If I guess 1, then 2, then 3, I could need 100 guesses. A smarter way: I guess 50, and you say "higher" or "lower". Now I know the answer is in half the range. I guess the middle of that half, and so on. After only 7 guesses, I always find it. It works because every answer ("higher" or "lower") cuts the possibilities in half.

**Example:** A dictionary is sorted by the alphabet. To find "monkey", you do not read from "aardvark". You open it near the middle, see "lemon", know that "monkey" comes later, and open the middle of the later half. A few flips and you are on the right page.

## 🔵 2. How it works
1. **Start with a sorted list** and look at the whole thing: the first position (low) to the last (high).
2. **Look at the middle item.**
3. **Compare it with what you want.**
   - If it is the one you want, you are done.
   - If the middle item is *smaller* than the target, the target must be in the *right* half. Ignore the left half and the middle.
   - If the middle item is *bigger*, the target must be in the *left* half. Ignore the right half and the middle.
4. **Repeat** on the half that is left: look at its middle, compare, throw half away.
5. **Stop** when you find it, or when nothing is left. If nothing is left, the item is not there.
6. **Count the looks.** Each look halves the number of places left, so the number of looks is about how many times you can halve the list size before nothing remains.

![[binary-search-depiction.svg]]
*A binary search for the number 7 in a sorted list of 17 numbers. The search first looks at the middle (position 8, the value 14), which is too big, so it moves left. Then it looks at position 3 (the value 6), too small, so it moves right. Then position 5 (the value 8), too big, and finally position 4 (the value 7): found, after 4 looks instead of up to 17. Image: AlwaysAngry, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Binary_Search_Depiction.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **It needs sorted data.** The reason it can throw half away is that everything left of the middle is smaller and everything right of it is bigger. If the list is not sorted, that reasoning is wrong, and the search can miss items that are there (the code shows this). Sorting costs time too (see [[Sorting]]), so binary search pays off when you search many times in the same sorted data.
- **It needs fast access by position:** looking at "item number 8" must be one step. A plain array or Python list is fine. A linked list is not (you would have to walk to the middle).
- **Cost:** at most $\lfloor\log_2 n\rfloor+1$ looks, which is $O(\log n)$ (equation 1 below). Compare linear search, which can need $n$ looks. For a million items, that is 20 against up to 1,000,000.
- **Compared with a hash table:** a hash table finds an exact key in about one step on average (see [[Hash Tables]]), but cannot tell you "the next larger value" or "everything between 10 and 20". Binary search on sorted data can do both.
- **Variants worth knowing:**
  - **Lower bound** (`bisect_left` in Python): the first position where a value could be inserted and the list stays sorted. If the value exists, it is the position of its first copy.
  - **Upper bound** (`bisect_right`): the position just after the last copy. The difference of the two counts how many copies there are (in the code).
  - **First "yes":** if a yes/no test is false up to some point and true from then on, binary search finds the point where it flips. Examples: "the first version that has the bug", or "the smallest $n$ with $n^2\ge1000$" (in the code).
  - **Bisection on a continuous range:** instead of a list, search an interval of real numbers for a place where a function changes sign (equation 3 below).
- **Python's tools:** the standard module `bisect` provides `bisect_left`, `bisect_right` and `insort`. They are written in a compiled language and are faster and safer than a hand-written loop (the code checks that my version agrees with `bisect_left`).
- **Common bugs:** off-by-one errors in the boundaries (`<` against `<=`, `mid` against `mid+1`), infinite loops when the range does not shrink, and midpoint overflow in languages with fixed-size integers (equation 2 below). A famous library bug of the last kind was reported in 2006 (from memory; I did not check the details).
- **Where it shows up:** looking up a value in a sorted column or index (see [[Pandas]]), finding a good setting by halving a range (for example, the largest batch size that fits in [[GPU]] memory, or a [[Learning Rate]] range; from memory, as common practice), and in [[Decision Tree]]s, where each question splits the data in two (a loose analogy: a tree's split is chosen to separate classes, not to halve the data evenly).
- **Limits:** the data must be sorted and stable during the search. If items are added or removed a lot, keeping the list sorted is costly, and a tree or a hash table may suit better (not covered here).

The code does seven small checks: (1) the search from the picture, with the positions it looks at; (2) the worst case for every list size; (3) lower and upper bounds against Python's `bisect`; (4) what happens on unsorted data; (5) the midpoint overflow with 32-bit numbers; (6) bisection for $\sqrt2$; (7) the first-true search.

```python
import bisect, math, warnings
import numpy as np
def search(a, target):                                   # returns (index or -1, list of positions looked at)
    lo, hi, looked = 0, len(a) - 1, []
    while lo <= hi:
        mid = lo + (hi - lo) // 2; looked.append(mid)    # lo + (hi-lo)//2, not (lo+hi)//2: see the overflow check below
        if a[mid] == target: return mid, looked
        if a[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1, looked
a = [1, 3, 4, 6, 7, 8, 10, 13, 14, 18, 19, 21, 24, 37, 40, 45, 71]
idx, looked = search(a, 7); print('find 7 in 17 numbers: index', idx, '| positions looked at', looked, '| values', [a[i] for i in looked])
print('find 5 (absent):', search(a, 5)[0], 'after', len(search(a, 5)[1]), 'looks')
# 1) the worst case is floor(log2 n) + 1 looks, for every size
ok = all(max(len(search(list(range(n)), t)[1]) for t in range(-1, n + 1)) == int(math.log2(n)) + 1 for n in range(1, 201))
print('worst case equals floor(log2 n)+1 for every n from 1 to 200:', ok, '| n=17 worst case:', max(len(search(a, t)[1]) for t in range(0, 80)))
# 2) the same idea gives the first place a sorted value could go (lower bound), and Python's bisect does it
def lower(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if a[mid] < x: lo = mid + 1
        else: hi = mid
    return lo
b = [1, 2, 2, 2, 5, 9]
print('lower bound of 2 and 3:', lower(b, 2), lower(b, 3), '| bisect gives', bisect.bisect_left(b, 2), bisect.bisect_left(b, 3), '| how many 2s:', bisect.bisect_right(b, 2) - bisect.bisect_left(b, 2))
rs = np.random.default_rng(0); data = sorted(rs.integers(0, 1000, 500).tolist())
print('lower() agrees with bisect_left on 2000 random queries:', all(lower(data, x) == bisect.bisect_left(data, x) for x in rs.integers(-5, 1005, 2000).tolist()))
# 3) it only works on sorted data
bad = [5, 1, 9, 3, 7]; print('unsorted [5,1,9,3,7]: search(5) ->', search(bad, 5)[0], '(found) but search(3) ->', search(bad, 3)[0], '(3 is at index 3, missed)')
# 4) the midpoint overflow in fixed-size integers (Python ints never overflow, so use 32-bit numbers to see it)
lo, hi = np.int32(1_500_000_000), np.int32(2_000_000_000)
with warnings.catch_warnings(): warnings.simplefilter('ignore'); bad_mid = (lo + hi) // np.int32(2)
print('32-bit (lo+hi)//2 =', int(bad_mid), '| lo+(hi-lo)//2 =', int(lo + (hi - lo) // np.int32(2)))
# 5) binary search on an answer: bisection for sqrt(2), and "first value where a yes/no test turns true"
def bisect_root(f, a, b, tol):
    steps = 0
    while b - a > tol:
        m = (a + b) / 2; steps += 1
        if f(a) * f(m) <= 0: b = m
        else: a = m
    return (a + b) / 2, steps
root, steps = bisect_root(lambda x: x * x - 2, 1.0, 2.0, 1e-6)
print('sqrt(2) by bisection:', round(root, 6), '(true', round(math.sqrt(2), 6), ') after', steps, 'halvings | formula ceil(log2((b-a)/tol)) =', math.ceil(math.log2(1 / 1e-6)))
def first_true(lo, hi, test):                              # smallest n in [lo, hi] with test(n) true, if test is false...false true...true
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if test(mid): hi = mid
        else: lo = mid + 1
    return lo
print('smallest n with n*n >= 1000:', first_true(0, 1000, lambda n: n * n >= 1000), '| check:', 31 * 31, 32 * 32)
# find 7 in 17 numbers: index 4 | positions looked at [8, 3, 5, 4] | values [14, 6, 8, 7]
# find 5 (absent): -1 after 4 looks
# worst case equals floor(log2 n)+1 for every n from 1 to 200: True | n=17 worst case: 5
# lower bound of 2 and 3: 1 4 | bisect gives 1 4 | how many 2s: 3
# lower() agrees with bisect_left on 2000 random queries: True
# unsorted [5,1,9,3,7]: search(5) -> 0 (found) but search(3) -> -1 (3 is at index 3, missed)
# 32-bit (lo+hi)//2 = -397483648 | lo+(hi-lo)//2 = 1750000000
# sqrt(2) by bisection: 1.414214 (true 1.414214 ) after 20 halvings | formula ceil(log2((b-a)/tol)) = 20
# smallest n with n*n >= 1000: 32 | check: 961 1024
```
Reading the output (Python 3.9; NumPy 1.24.4 only for the 32-bit numbers and random data):
- **The picture's example:** searching for 7 among 17 numbers looks at positions 8, 3, 5 and 4, which hold the values 14, 6, 8 and 7. That is 4 looks, the same trail as in the figure. Searching for 5, which is not there, ends after 4 looks with no match.
- **Worst case:** for every list size from 1 to 200, the longest search over all possible targets (present or absent) is exactly $\lfloor\log_2n\rfloor+1$ looks. For the 17-number list the worst case is 5.
- **Lower and upper bounds:** in the list `[1, 2, 2, 2, 5, 9]` the first place for 2 is position 1 and for 3 is position 4, and the number of 2s is 3. Python's `bisect` agrees, and so does my version on 2,000 random queries.
- **Unsorted data:** in `[5, 1, 9, 3, 7]` the search finds 5 only by luck (it happens to be the first position it checks after one step), but it misses 3 even though 3 is in the list.
- **Overflow:** with 32-bit whole numbers, adding $1{,}500{,}000{,}000$ and $2{,}000{,}000{,}000$ goes past the largest value ($2{,}147{,}483{,}647$) and wraps around to a negative number, so `(lo+hi)//2` gives $-397{,}483{,}648$. The safe form `lo+(hi-lo)//2` gives $1{,}750{,}000{,}000$. (Python's own integers have no limit, so ordinary Python code does not have this bug.)
- **Bisection:** halving the interval $[1,2]$ until it is narrower than $10^{-6}$ takes exactly 20 halvings and gives $1.414214$, matching $\sqrt2$.
- **First true:** the smallest $n$ whose square is at least 1,000 is 32 ($31^2=961$, $32^2=1024$).
- **Limits:** these are tiny lists, and I counted looks, not time. I did not time binary search against linear search.

## 🔴 4. Deep / formal

**Why the looks are logarithmic.** Let $s$ be the number of positions still possible. One look at the middle leaves at most $\lfloor s/2\rfloor$ (it discards the middle and one whole side). Starting from $n$, after $k$ looks at most $\lfloor n/2^k\rfloor$ remain, and the search ends when that reaches 0. So in the worst case

$$\text{looks}=\lfloor\log_2 n\rfloor+1,\qquad s_{k}\le\Big\lfloor\frac{n}{2^{k}}\Big\rfloor$$

**Where:**
- $n$ is the number of items. **Matters:** doubling $n$ adds only one look, because one extra look halves the doubled list back to its old size.
- $s_k$ is how many positions are still possible after $k$ looks. **Matters:** it shrinks by at least half each time, which is the whole reason the method is fast.
- $\lfloor\cdot\rfloor$ ("floor") rounds down. **Matters:** the sizes are whole numbers of positions, and the middle item is removed from the part that is left, so a list of size $s$ leaves at most $\lfloor s/2\rfloor$.
- $\log_2n$ is how many times you can halve $n$. **Matters:** it grows very slowly: a million items need about 20 looks, and a trillion about 40.
- The $+1$ counts the final look, taken when one item is left. **Matters:** without it the formula would be short by one in the worst case.

**Example (tiny numbers):** $n=17$ (the picture's list).
1. Positions still possible: $17\to8\to4\to2\to1\to0$.
2. That is 5 looks in the worst case. The formula gives $\lfloor\log_2 17\rfloor+1=\lfloor4.09\rfloor+1=5$.
3. The code found that the longest search over all targets for this list is 5. The search for 7 happened to be lucky and finished in 4.
4. Meaning: for 17 items you never need more than 5 looks, and the picture's 4 is a typical lucky case. For $n=1000$ it is 10 looks and for $n=10^6$ it is 20.

**Taking the middle safely.** With low position $lo$ and high position $hi$ (both whole numbers), the middle is

$$mid=lo+\Big\lfloor\frac{hi-lo}{2}\Big\rfloor\quad\text{instead of}\quad mid=\Big\lfloor\frac{lo+hi}{2}\Big\rfloor$$

**Where:**
- $lo$ and $hi$ are the ends of the part still possible. **Matters:** both can be large, up to the size of the list.
- $hi-lo$ is the width. **Matters:** it is never larger than $hi$, so it cannot overflow when $lo$ and $hi$ fit.
- $\lfloor\cdot\rfloor$ rounds down. **Matters:** the two forms give the same answer in exact arithmetic, so the safe one changes nothing except avoiding the overflow.
- The sum $lo+hi$ in the first form. **Matters:** in a language with fixed-size integers, it can go past the largest value and wrap around to a negative number, giving a bad middle and a crash or wrong answer. Python's integers are unlimited, so this is a problem in other languages (and in NumPy).

**Example (tiny numbers):** 32-bit whole numbers hold values up to $2{,}147{,}483{,}647$. Take $lo=1{,}500{,}000{,}000$ and $hi=2{,}000{,}000{,}000$.
1. Sum: $3{,}500{,}000{,}000$, which is more than the largest value, so it wraps to $3{,}500{,}000{,}000-2^{32}=-794{,}967{,}296$.
2. Unsafe middle: $-794{,}967{,}296/2=-397{,}483{,}648$. That is a negative position, which is wrong. The code printed this number.
3. Safe middle: $hi-lo=500{,}000{,}000$, half is $250{,}000{,}000$, and $1{,}500{,}000{,}000+250{,}000{,}000=1{,}750{,}000{,}000$. Correct, and the code printed it too.
4. Meaning: the two formulas are the same on paper but not on a machine with limited whole numbers. Writing $lo+(hi-lo)/2$ costs nothing and avoids the bug.

![[bisection-method.svg]]
*The bisection method, which is binary search on a continuous range. The curve $F(x)$ is positive at $a_1$ and negative at $b_1$, so it must cross zero somewhere between them (the red dot). Take the middle of the interval, see which side has the sign change, and keep that half. The braces at the bottom show the interval halving again and again, closing in on the crossing. Image: Tokuchan, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bisection_method.svg), CC BY-SA 3.0.*

**Bisection to a tolerance.** To find a root of a continuous function that changes sign on $[a,b]$, halve the interval each time. After $k$ halvings the interval has width $(b-a)/2^k$, so to reach a width of at most $\varepsilon$ you need

$$k=\Big\lceil\log_2\frac{b-a}{\varepsilon}\Big\rceil\ \text{halvings}$$

**Where:**
- $[a,b]$ is the starting interval, with the function's sign different at the two ends. **Matters:** the sign change guarantees a root in between (for a continuous function). Without it there may be none to find.
- $\varepsilon$ ("epsilon") is how narrow you want the final interval. **Matters:** a smaller $\varepsilon$ means a more precise answer but more halvings. Each extra decimal digit of accuracy costs about 3.3 more halvings.
- $\log_2\frac{b-a}{\varepsilon}$ is how many times you can halve the width before it drops below $\varepsilon$. **Matters:** the cost depends on the *ratio* of the starting width to the target width, not on the function.
- $\lceil\cdot\rceil$ ("ceiling") rounds up. **Matters:** you cannot do a fraction of a halving, so you must do the next whole one.

**Example (tiny numbers):** Find $\sqrt2$ as the root of $x^2-2$ on $[1,2]$ (the function is $-1$ at 1 and $+2$ at 2), with $\varepsilon=10^{-6}$.
1. Starting width $b-a=1$. Ratio $1/10^{-6}=10^6$.
2. $\log_2 10^6=19.93$, so $k=\lceil19.93\rceil=20$ halvings.
3. After 20 halvings the width is $1/2^{20}=9.5\times10^{-7}<10^{-6}$.
4. The code made exactly 20 halvings and gave $1.414214$.
5. Meaning: each halving gains a bit of accuracy, and 20 of them are enough for six decimal places. It is slow compared with fancier root-finders (from memory, such as Newton's method), but it never fails when the sign change holds.

**Other points (from memory, not checked here):**
- Binary search was described in the 1940s and early programmers made many mistakes writing it. Knuth is often quoted on how long it took before a correct version was published (a claim I did not check, and I have not tried to recall the exact numbers).
- A related idea is *exponential search* (double a step until you overshoot, then binary search back), which helps when the list has no known end. I have not covered or tested it.
- In machine learning, binary search appears inside several tools, for example in some ways of finding the right threshold for a classifier (see [[Evaluation Metrics]]). I did not check any particular library.
- I tested only whole-number lists and one function. Lists of text, lists with many equal items under a custom order, and floating-point edge cases are not covered.

**What came before, and how this improved on it:**
- **Before:** scan the list from the start until the item is found (linear search).
- **Its limit:** the work grows with the length of the list, and a million items take about half a million looks on average.
- **What changed:** if the list is sorted, look at the middle and throw away the wrong half. The idea was described in the 1940s and 1950s, and a correct version took years to appear in print (Knuth says so; from memory, not checked).
- **How much better:** in the code below, 200 random lookups in a sorted list of 1,000,000 numbers took 531,955 looks on average with a scan (995,901 at most) and 18.9 with binary search (20 at most). That is about 28,000 times fewer on average. The price is that the list must already be sorted.

**Check it yourself:**

```python
import random
n = 1_000_000; a = list(range(0, 2 * n, 2)); random.seed(0)
def linear(x):
    for i, v in enumerate(a):
        if v == x: return i, i + 1
def binary(x):
    lo, hi, c = 0, len(a) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2; c += 1
        if a[mid] == x: return mid, c
        lo, hi = (mid + 1, hi) if a[mid] < x else (lo, mid - 1)
targets = random.sample(a, 200)
lc = [linear(x)[1] for x in targets]; bc = [binary(x)[1] for x in targets]
print(f"sorted list of {n:,} numbers, 200 random lookups")
print(f"  scan from the start: {sum(lc) / 200:,.0f} looks on average, {max(lc):,} at most")
print(f"  binary search:       {sum(bc) / 200:.1f} looks on average, {max(bc)} at most (log2 of n = {n.bit_length()})")
```

## Related
[[Big-O]] · [[Hash Tables]] · [[Sorting]] · [[Python]] · [[Pandas]] · [[NumPy]] · [[Decision Tree]] · [[Learning Rate]] · [[GPU]] · [[Evaluation Metrics]] · [[Recursion]] · [[Arrays and Lists]] · [[Trees]] · [[SQL]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, the exercises and sections on searching (from memory; I did not open it for this note).
- Knuth, *The Art of Computer Programming*, volume 3, on searching (from memory; I did not open it).
- Python documentation for the `bisect` module (docs.python.org).
- Figures: [Binary_Search_Depiction.svg](https://commons.wikimedia.org/wiki/File:Binary_Search_Depiction.svg) (AlwaysAngry, CC BY-SA 4.0) and [Bisection_method.svg](https://commons.wikimedia.org/wiki/File:Bisection_method.svg) (Tokuchan, CC BY-SA 3.0), both on Wikimedia Commons. I viewed both before writing the captions.
- The code and every Example number were run for this note (Python 3.9, NumPy 1.24.4).

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/searching-sorting Binary search:::Guess the middle of a sorted list, learn "higher" or "lower", and throw half away. Repeat on the half that is left.

#flashcards/algorithms/concept #flashcards/topic/searching-sorting List the steps of binary search.::Start with a sorted list (low to high). Look at the middle item and compare it with the target. If it is smaller, keep the right half. If it is bigger, keep the left half. Repeat. Stop when found, or when nothing is left (the item is not there).

#flashcards/algorithms/formula #flashcards/topic/searching-sorting Write the number of looks and the shrinking list and explain each symbol.
?
$$\text{looks}=\lfloor\log_2 n\rfloor+1,\qquad s_{k}\le\Big\lfloor\frac{n}{2^{k}}\Big\rfloor$$
- $n$: the number of items. Doubling $n$ adds only one look, because one extra look halves the doubled list back to its old size.
- $s_k$: how many positions are still possible after $k$ looks. It shrinks by at least half each time, which is why the method is fast.
- $\lfloor\cdot\rfloor$ (floor): the sizes are whole numbers, and the middle item is removed from the part left, so a list of size $s$ leaves at most $\lfloor s/2\rfloor$.

#flashcards/algorithms/example #flashcards/topic/searching-sorting $n=17$. How many looks at most, and what happened in the picture?::Positions still possible: $17\to8\to4\to2\to1\to0$, so 5 looks in the worst case ($\lfloor\log_2 17\rfloor+1=5$). The code found 5 as the longest search over all targets. The search for 7 was lucky and finished in 4. For $n=1000$ it is 10 looks and for $n=10^6$ it is 20.

#flashcards/algorithms/formula #flashcards/topic/searching-sorting Write the safe middle of a search range and explain each symbol.
?
$$mid=lo+\Big\lfloor\frac{hi-lo}{2}\Big\rfloor\quad\text{instead of}\quad mid=\Big\lfloor\frac{lo+hi}{2}\Big\rfloor$$
- $lo$, $hi$: the ends of the part still possible. Both can be large, up to the size of the list.
- $hi-lo$: the width. It is never larger than $hi$, so it cannot overflow when $lo$ and $hi$ fit.
- $\lfloor\cdot\rfloor$ rounds down. In exact arithmetic the two forms give the same answer, so the safe one changes nothing except avoiding overflow.
- The sum $lo+hi$ in the first form can pass the largest value of a fixed-size integer and wrap around to a negative number.

#flashcards/algorithms/example #flashcards/topic/searching-sorting 32-bit integers hold values up to $2{,}147{,}483{,}647$. With $lo=1{,}500{,}000{,}000$ and $hi=2{,}000{,}000{,}000$, what do the two middles give?::Sum $3{,}500{,}000{,}000$ wraps to $3{,}500{,}000{,}000-2^{32}=-794{,}967{,}296$. Unsafe middle $-397{,}483{,}648$, a negative position, which is wrong (the code printed this). Safe: $hi-lo=500{,}000{,}000$, half $250{,}000{,}000$, so $1{,}750{,}000{,}000$, correct.

#flashcards/algorithms/formula #flashcards/topic/searching-sorting Write the number of halvings of bisection and explain each symbol.
?
$$k=\Big\lceil\log_2\frac{b-a}{\varepsilon}\Big\rceil\ \text{halvings}$$
- $[a,b]$: the starting interval, with the function's sign different at the two ends. The sign change guarantees a root in between (for a continuous function).
- $\varepsilon$: how narrow you want the final interval. A smaller $\varepsilon$ means more precision but more halvings. Each extra decimal digit costs about 3.3 more halvings.
- $\log_2\frac{b-a}{\varepsilon}$: how many times you can halve the width before it drops below $\varepsilon$.

#flashcards/algorithms/example #flashcards/topic/searching-sorting Find $\sqrt2$ as the root of $x^2-2$ on $[1,2]$ with $\varepsilon=10^{-6}$. How many halvings?::Width 1, ratio $10^6$. $\log_2 10^6=19.93$, so $k=20$. After 20 halvings the width is $1/2^{20}=9.5\times10^{-7}<10^{-6}$. The code made exactly 20 halvings and gave 1.414214. It is slow compared with fancier root-finders (from memory).

#flashcards/algorithms/image #flashcards/topic/searching-sorting ![[binary-search-depiction.svg]]
What does this picture show?
?
A binary search for 7 in a sorted list of 17 numbers. It first looks at the middle (position 8, value 14), too big, so it moves left. Then position 3 (value 6), too small, so it moves right. Then position 5 (value 8), too big, and finally position 4 (value 7): found after 4 looks instead of up to 17. (From the note's caption: AlwaysAngry, Wikimedia Commons.)

#flashcards/algorithms/image #flashcards/topic/searching-sorting ![[bisection-method.svg]]
What does this picture show?
?
The bisection method, which is binary search on a continuous range. The curve $F(x)$ is positive at $a_1$ and negative at $b_1$, so it must cross zero between them (the red dot). Take the middle, see which side has the sign change, and keep that half. The braces at the bottom show the interval halving again and again. (From the note's caption: Tokuchan, Wikimedia Commons.)

#flashcards/algorithms/cloze #flashcards/topic/searching-sorting Binary search needs a ==sorted== list.
