---
tags: [algorithms, greedy, huffman, compression, optimization, programming, python]
status: growing
level: 4
tier: 2
---

# Greedy Algorithms

A **greedy algorithm** builds an answer one step at a time, and at every step it takes **whatever looks best right now**, without ever looking back or planning ahead. It is fast and simple. For some problems this short-sighted approach gives the perfect answer. For others it gives a bad answer. The skill is knowing which problems are which. [[Dijkstra's Algorithm]] (always finish the closest place) and Huffman coding (always merge the two rarest symbols) are two famous greedy methods.

## 🟢 1. Simple (age 12)
You are given a big pile of coins and asked to hand over exactly 36 cents using as few coins as possible. A natural way: take the biggest coin that still fits (20), then the biggest that fits in what is left (10), then 5, then 1. That is four coins, and it really is the fewest. Grabbing the best-looking option each time is being "greedy".

But it does not always work. Imagine a strange country with coins worth 1, 3 and 4. To pay 6, the greedy way takes a 4 and then two 1s: three coins. A smarter way uses two 3s: only two coins. Being greedy cost an extra coin.

**Example:** Packing a bag by always grabbing the most valuable thing first can be a trap. You might use up all the room on one big item, when two smaller ones together would have been worth more.

## 🔵 2. How it works
1. **Pick a rule for "best right now".** For coins it is "the biggest coin that fits". For scheduling jobs it might be "the job that ends first".
2. **Apply the rule, commit to it, and never undo it.** Make the choice, shrink the problem, and repeat on what is left.
3. **Stop** when nothing is left to decide.
4. **Check whether the rule is safe.** For a problem where greedy works, you can show two things. First, the greedy choice can always be part of *some* best answer (the *greedy-choice property*). Second, what remains after the choice is a smaller copy of the same problem (*optimal substructure*, the same idea as in [[Dynamic Programming]]).
5. **If you cannot show it, test it.** Compare with a slower method that is always right (trying every possibility, or dynamic programming) on small cases. If the greedy answer is ever worse, the rule is not safe.

![[greedy-coins-36-cents.svg]]
*Making 36 cents greedily: take the biggest coin that fits each time. 36 minus 20 is 16, 16 minus 10 is 6, 6 minus 5 is 1, 1 minus 1 is 0, so the coins are 20, 10, 5 and 1. For this set of coins, the code finds that greedy is the fewest for every amount from 1 to 100. Image: Nandhp, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Greedy_algorithm_36_cents.svg), public domain.*

![[huffman-coding-visualisation.svg]]
*Huffman coding of the message "A_DEAD_DAD_CEDED_A_BAD_BABE_A_BEADED_ABACA_BED" (the underscore is a space). In steps 2 to 6 the two rarest items are merged into one, again and again, building a tree. Step 7 lists the final codes (A is 10, B is 1111, C is 1110, D is 01, E is 110, the underscore is 00), and step 8 is the whole message in bits. The code below rebuilds the same code lengths. Image: Cmglee, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Huffman_coding_visualisation.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **When greedy works:** it is guaranteed for problems with the greedy-choice property and optimal substructure. Standard examples are *activity selection* (choose the most non-overlapping jobs), *fractional knapsack*, *Huffman coding*, and shortest paths with non-negative lengths ([[Dijkstra's Algorithm]]).
- **Coins:** for the coin set 25, 10, 5, 1, greedy was the best for every amount from 1 to 100 in the code. For the set 4, 3, 1, it was *not* the best for 24 of the 100 amounts, starting at 6, 10, 14, 18, 22. Whether a given coin set is "greedy-safe" depends on the coins (from memory, there are tests for it; I did not look any up or use one).
- **Activity selection:** sort the jobs by finishing time and take each job that starts after the last chosen one ends. On 300 random job sets, this earliest-finish rule matched the best answer (found by trying every subset) all 300 times. Other plausible rules fail: "earliest start" matched 272 times, "shortest job first" only 154. A small counterexample for shortest-first: jobs (0,4), (3,5), (4,8). The shortest job is (3,5), which blocks both others, so greedy picks 1 job where 2 are possible.
- **Knapsack:** if you may take *fractions* of items, greedy by value per weight is exact. With whole items it is not: with items (60, weight 10), (100, weight 20), (120, weight 30) and capacity 50, greedy takes the first two for 160, and the best is 220 (the second and third). With fractions allowed, greedy gets 240.
- **Huffman coding:** a way to store text in fewer bits, by giving **short codes to common symbols and long codes to rare ones**. Greedy rule: repeatedly merge the two least frequent items. The merging builds a binary tree (see [[Trees]]), where each symbol's code is the path of left and right turns from the root. The code uses a heap (see [[Heaps and Priority Queues]]) to find the two lightest items quickly. Equation 1 to 3 below make this exact.
- **Prefix-free:** no code is the start of another, so a stream of bits can be decoded without gaps or separators (the code checks this, and decodes a message back correctly).
- **Cost** (see [[Big-O]]): Huffman with a heap is $O(n\log n)$ for $n$ symbols. Activity selection is dominated by sorting, so also $O(n\log n)$ (see [[Sorting]]).
- **In this vault:** Huffman's bound is stated in terms of [[Entropy]] (equation 3). A [[Decision Tree]] is usually built greedily too: each split is chosen to look best at that moment, without looking ahead (that is from memory about how common implementations work; I did not check a specific library).
- **Limits:** the code runs small cases only. It does not test every possible greedy rule, does not prove any rule correct, and does not time anything.

The code covers five things: (1) coin change, greedy against the best answer, (2) activity selection with three rules, (3) knapsack with fractions and with whole items, (4) Huffman coding on two examples, with sizes, entropy and a round trip, (5) Huffman against a check of every merge order.

```python
import heapq, itertools, math, random
from functools import lru_cache
# 1) coin change: always take the biggest coin that fits. Right for some coin sets, wrong for others
def greedy_coins(coins, amount):
    n = 0
    for c in sorted(coins, reverse=True): n += amount // c; amount %= c
    return n
def best_coins(coins, amount):                                       # dynamic programming answer (see Dynamic Programming)
    dp = [0] + [10 ** 9] * amount
    for a in range(1, amount + 1): dp[a] = min(dp[a - c] + 1 for c in coins if c <= a)
    return dp[amount]
for coins in ([25, 10, 5, 1], [4, 3, 1]):
    bad = [a for a in range(1, 101) if greedy_coins(coins, a) != best_coins(coins, a)]
    print('coins', coins, '| amounts 1-100 where greedy is not the best:', len(bad), '| first few:', bad[:5])
print('picture: 36 with coins [20,10,5,1] -> greedy', greedy_coins([20, 10, 5, 1], 36), 'coins | best', best_coins([20, 10, 5, 1], 36), '| amounts 1-100 where greedy fails for this set:', sum(greedy_coins([20, 10, 5, 1], a) != best_coins([20, 10, 5, 1], a) for a in range(1, 101)))
print('amount 6 with coins [4,3,1]: greedy', greedy_coins([4, 3, 1], 6), 'coins (4+1+1) | best', best_coins([4, 3, 1], 6), 'coins (3+3)')
# 2) activity selection: pick the job that ends earliest. Compare with trying every subset, and with two other rules
def pick(jobs, key):
    chosen, end = 0, -1
    for s, e in sorted(jobs, key=key):
        if s >= end: chosen += 1; end = e
    return chosen
def best_jobs(jobs):
    best = 0
    for r in range(len(jobs) + 1):
        for sub in itertools.combinations(sorted(jobs), r):
            if all(sub[i][1] <= sub[i + 1][0] for i in range(len(sub) - 1)): best = max(best, r)
    return best
rng = random.Random(4); ok = {'earliest end': 0, 'earliest start': 0, 'shortest first': 0}
for _ in range(300):
    jobs = [(s, s + rng.randint(1, 6)) for s in (rng.randint(0, 12) for _ in range(rng.randint(1, 9)))]
    b = best_jobs(jobs)
    ok['earliest end'] += pick(jobs, lambda j: (j[1], j[0])) == b; ok['earliest start'] += pick(jobs, lambda j: (j[0], j[1])) == b; ok['shortest first'] += pick(jobs, lambda j: (j[1] - j[0], j[0])) == b
print('300 random job sets, greedy matches the best answer:', ok)
print('shortest-first fails here:', pick([(0, 4), (3, 5), (4, 8)], lambda j: (j[1] - j[0], j[0])), 'jobs chosen, best is', best_jobs([(0, 4), (3, 5), (4, 8)]))
# 3) knapsack: greedy by value per weight is exact if you may take fractions, but not for whole items
items, cap = [(60, 10), (100, 20), (120, 30)], 50
left, frac, whole = cap, 0.0, 0
for v, w in sorted(items, key=lambda x: -x[0] / x[1]):
    take = min(1, left / w); frac += v * take
    if w <= left: whole += v
    left -= w * take
print('knapsack capacity 50: fractions allowed, greedy gets', frac, '| whole items only, greedy gets', whole, '| best for whole items', max(sum(v for v, _ in c) for r in range(4) for c in itertools.combinations(items, r) if sum(w for _, w in c) <= cap))
# 4) Huffman coding: repeatedly merge the two lightest items. Short codes for common symbols
def huffman(freq):
    heap = [(f, i, s) for i, (s, f) in enumerate(freq.items())]; heapq.heapify(heap); n = len(heap); merges = 0
    while len(heap) > 1:
        (f1, _, a), (f2, _, b) = heapq.heappop(heap), heapq.heappop(heap); merges += f1 + f2
        heapq.heappush(heap, (f1 + f2, n, (a, b))); n += 1
    codes = {}
    def walk(t, c):
        if isinstance(t, tuple): walk(t[0], c + '0'); walk(t[1], c + '1')
        else: codes[t] = c or '0'
    walk(heap[0][2], ''); return codes, merges
freq = {'a': 45, 'b': 13, 'c': 12, 'd': 16, 'e': 9, 'f': 5}
codes, merges = huffman(freq); total = sum(freq.values())
cost = sum(freq[s] * len(c) for s, c in codes.items())
print('code lengths:', {s: len(c) for s, c in sorted(codes.items())}, '| weighted length', cost, '= sum of merge weights', merges, '| fixed 3-bit code would cost', 3 * total)
H = -sum(f / total * math.log2(f / total) for f in freq.values()); L = cost / total
print('average bits per symbol L =', round(L, 3), '| entropy H =', round(H, 3), '| H <= L < H + 1:', H <= L < H + 1, '| Kraft sum', sum(2 ** -len(c) for c in codes.values()))
msg = 'abacabadfeabcdea'; enc = ''.join(codes[ch] for ch in msg); inv = {c: s for s, c in codes.items()}
dec, cur = '', ''
for bit in enc:
    cur += bit
    if cur in inv: dec += inv[cur]; cur = ''
print('round trip works:', dec == msg, '| prefix-free:', not any(a != b and b.startswith(a) for a in codes.values() for b in codes.values()), '| message bits', len(enc), 'against', 3 * len(msg), 'fixed')
msg2 = 'A_DEAD_DAD_CEDED_A_BAD_BABE_A_BEADED_ABACA_BED'              # the message in the second picture
f2 = {ch: msg2.count(ch) for ch in sorted(set(msg2))}; c2, m2 = huffman(f2)
print('second picture: counts', f2, '| code lengths', {s: len(c) for s, c in sorted(c2.items())}, '| total bits', sum(f2[s] * len(c) for s, c in c2.items()), '| fixed 3-bit', 3 * len(msg2))
@lru_cache(None)
def best_merge(ws):                                                   # try every pair to merge: the true minimum total merge cost
    if len(ws) == 1: return 0
    return min(ws[i] + ws[j] + best_merge(tuple(sorted(ws[:i] + ws[i + 1:j] + ws[j + 1:] + (ws[i] + ws[j],)))) for i in range(len(ws)) for j in range(i + 1, len(ws)))
same = 0
for _ in range(200):
    w = tuple(sorted(rng.randint(1, 20) for _ in range(rng.randint(2, 6)))); same += huffman({str(i): x for i, x in enumerate(w)})[1] == best_merge(w)
print('random weight sets (2 to 6 symbols): Huffman equals the best of every merge order in', same, 'of 200')
# coins [25, 10, 5, 1] | amounts 1-100 where greedy is not the best: 0 | first few: []
# coins [4, 3, 1] | amounts 1-100 where greedy is not the best: 24 | first few: [6, 10, 14, 18, 22]
# picture: 36 with coins [20,10,5,1] -> greedy 4 coins | best 4 | amounts 1-100 where greedy fails for this set: 0
# amount 6 with coins [4,3,1]: greedy 3 coins (4+1+1) | best 2 coins (3+3)
# 300 random job sets, greedy matches the best answer: {'earliest end': 300, 'earliest start': 272, 'shortest first': 154}
# shortest-first fails here: 1 jobs chosen, best is 2
# knapsack capacity 50: fractions allowed, greedy gets 240.0 | whole items only, greedy gets 160 | best for whole items 220
# code lengths: {'a': 1, 'b': 3, 'c': 3, 'd': 3, 'e': 4, 'f': 4} | weighted length 224 = sum of merge weights 224 | fixed 3-bit code would cost 300
# average bits per symbol L = 2.24 | entropy H = 2.22 | H <= L < H + 1: True | Kraft sum 1.0
# round trip works: True | prefix-free: True | message bits 39 against 48 fixed
# second picture: counts {'A': 11, 'B': 6, 'C': 2, 'D': 10, 'E': 7, '_': 10} | code lengths {'A': 2, 'B': 4, 'C': 4, 'D': 2, 'E': 3, '_': 2} | total bits 115 | fixed 3-bit 138
# random weight sets (2 to 6 symbols): Huffman equals the best of every merge order in 200 of 200
```
Reading the output (Python 3.9, standard library only):
- **Coins:** the 25, 10, 5, 1 set never loses to the best answer from 1 to 100. The 4, 3, 1 set loses at 24 amounts, the first being 6 (greedy 4+1+1 is 3 coins, best 3+3 is 2). The picture's set 20, 10, 5, 1 gives 4 coins for 36 and never loses from 1 to 100.
- **Jobs:** earliest-finish matched the best answer in 300 of 300 random cases, earliest-start in 272, shortest-first in 154. On the counterexample, shortest-first picks 1 job and the best is 2.
- **Knapsack:** 240 with fractions, 160 for whole items by greedy, 220 for the true best.
- **Huffman, first example:** frequencies 45, 13, 12, 16, 9, 5 per 100 symbols. Code lengths are 1, 3, 3, 3, 4, 4. The weighted length is 224 bits per 100 symbols, the same as the sum of the merge weights, against 300 for a fixed 3-bit code. The average is $L=2.24$ bits per symbol and the entropy is $H=2.22$. The condition $H\le L<H+1$ holds, and the Kraft sum is 1. A test message decodes back correctly with a prefix-free code, and takes 39 bits instead of 48.
- **Huffman, second example (the Commons picture's message):** the symbol counts are 11, 6, 2, 10, 7, 10, and the code lengths are 2, 4, 4, 2, 3, 2 (A, B, C, D, E and the underscore). That is the same set of lengths as in the picture. The total is 115 bits against 138 for 3 bits each.
- **Optimal check:** for 200 random sets of 2 to 6 weights, Huffman's total cost equals the best of every possible merge order in all 200.
- **Limits:** small inputs, and nothing timed.

## 🔴 4. Deep / formal

**The cost of a code.** Give each symbol $i$ a bit string of length $l_i$, and let $f_i$ be how often the symbol occurs. The total number of bits is the *weighted length*. In a Huffman tree it also equals the sum of the weights of all the merged (internal) nodes:

$$B=\sum_i f_i\,l_i\ =\ \sum_{\text{merges}}(w_x+w_y)$$

**Where:**
- $f_i$ is the frequency of symbol $i$ (a count, or a fraction of the whole). **Matters:** common symbols should get short codes, because their length is multiplied by a large $f_i$.
- $l_i$ is the length of its code, which is the same as its depth in the tree (the number of left or right turns from the root). **Matters:** this is what the algorithm tries to keep small where $f_i$ is big.
- $w_x+w_y$ is the combined weight of the two items merged at one step. **Matters:** each merge pushes every symbol under it one level deeper, so each symbol's weight is counted once for every merge above it, which is the same as its depth. That is why the two sums are equal. This is a short counting argument that I checked numerically.
- The greedy rule merges the two *smallest* weights each time. **Matters:** merging small weights early keeps their extra depth cheap. Merging big weights first would make large weights pay for many levels.

**Example (tiny numbers):** frequencies 45, 13, 12, 16, 9, 5 (per 100 symbols).
1. Merge the two smallest: $5+9=14$. Then $12+13=25$. Then $14+16=30$. Then $25+30=55$. Then $45+55=100$.
2. Sum of merge weights: $14+25+30+55+100=224$.
3. By lengths: the code lengths are 1, 3, 3, 3, 4, 4 for the symbols a, b, c, d, e, f, so $B=45\cdot1+13\cdot3+12\cdot3+16\cdot3+9\cdot4+5\cdot4=45+39+36+48+36+20=224$. The code also gave 224.
4. Meaning: 100 symbols take 224 bits, against 300 with a fixed 3-bit code, a saving of about 25%.

**Which code lengths are possible?** A set of code lengths $l_1,\dots,l_n$ can be the lengths of a prefix-free binary code exactly when:

$$\sum_{i=1}^{n}2^{-l_i}\ \le\ 1$$

**Where:**
- $l_i$ is the length of the code for symbol $i$. **Matters:** every extra bit halves the share of "code space" a symbol uses.
- $2^{-l_i}$ is the share of all possible bit strings that start with symbol $i$'s code. **Matters:** a code of length 1 blocks half of everything (all strings starting with that bit), and a code of length 3 blocks an eighth. In a prefix-free code, no two codes may block the same strings.
- The sum $\le1$ means the shares fit in the whole. **Matters:** if the sum were above 1, the codes would overlap, so some code would be the start of another and decoding would be ambiguous.
- A sum of exactly 1 means no code space is wasted. **Matters:** a Huffman code is *full* in this sense. I am quoting the "exactly when" from the standard result (from memory); the code only checks that this one set has sum 1.

**Example (tiny numbers):** the lengths 1, 3, 3, 3, 4, 4 from the first example.
1. $2^{-1}=0.5$, three times $2^{-3}=0.125$ is $0.375$, and two times $2^{-4}=0.0625$ is $0.125$.
2. Total: $0.5+0.375+0.125=1$. The code printed 1.0.
3. Meaning: the codes use up all of the available code space with no overlap. It is a valid prefix-free code with none of the space wasted.

**How good can a code be?** Let $p_i=f_i/\sum f$ be the probability of symbol $i$. The *entropy* $H=-\sum p_i\log_2p_i$ (see [[Entropy]]) is the average number of bits per symbol that no code can beat. The average length of a Huffman code $L=\sum p_il_i$ satisfies:

$$H\ \le\ L\ <\ H+1$$

**Where:**
- $H$ is the entropy in bits per symbol. **Matters:** it is the lower limit: the more predictable the text, the smaller $H$ is, and the more it can be squeezed.
- $L$ is the average Huffman code length, in bits per symbol. **Matters:** it is what you actually pay. It cannot go below $H$ for any prefix-free code.
- The gap of less than 1 bit per symbol is the price of using whole bits. **Matters:** Huffman cannot split a bit. When one symbol is very likely (probability near 1), $H$ is tiny but every symbol still needs at least 1 bit, so the gap is large in proportion.
- The condition holds for the code in this note. **Matters:** the statement is a standard result that I am quoting from memory and tested on two examples, but did not prove.

**Example (tiny numbers):** the first example, probabilities 0.45, 0.13, 0.12, 0.16, 0.09, 0.05.
1. Entropy: $H=0.45\cdot1.152+0.13\cdot2.943+0.12\cdot3.059+0.16\cdot2.644+0.09\cdot3.474+0.05\cdot4.322\approx2.22$ bits.
2. Average code length: $L=224/100=2.24$ bits.
3. Check: $2.22\le2.24<3.22$. True.
4. Meaning: the Huffman code is within 0.02 bits per symbol of the theoretical limit here, and a fixed 3-bit code is 0.78 above it.

**Why the greedy rules above are safe or not (a sketch, not a proof).** For activity selection, take any best schedule. If its first job is not the one that ends earliest, swap it for the one that ends earliest: that job ends no later, so it still fits with the rest, and the count does not drop. This *exchange argument* shows the greedy first choice is part of some best answer, and the rest of the schedule is the same problem on the jobs after it. The same pattern (swap the best-looking choice into an optimal answer without making it worse) is the standard proof for Huffman and for the other safe cases. I checked these only on small cases by running them, and I did not write full proofs.

**Other points (from memory, not checked here):**
- Huffman published his algorithm in 1952, while a student. I did not check the date, and I did not check the story that it came from a class assignment.
- Real compressors usually combine Huffman-style codes with other steps (for example a dictionary method first). I did not check how any particular format does it.
- Greedy methods also appear in minimum spanning trees (Kruskal's and Prim's algorithms). I did not cover or run them.
- A coin set where greedy is always optimal is called *canonical*. I did not check the exact name or a test for it.
- Greedy is often used as a fast approximation when the exact answer is too slow, with guarantees for some problems. I did not look into any.

**What came before, and how this improved on it:**
- **Before:** try every combination and pick the best one.
- **Its limit:** the number of combinations explodes.
- **What changed:** always take the best-looking choice now and never undo it. It is fast, but it is only correct for some problems.
- **How much better:** in the code below, making 41 cents from US coins, greedy used 4 steps and got 4 coins, while trying everything made 465,934 calls to find the same answer. **Honest catch:** with coins 1, 3 and 4 and an amount of 6, greedy gave 3 coins (4+1+1) while the best is 2 (3+3). The speed is real, and so is the risk of a wrong answer; [[Dynamic Programming]] repairs that case.

**Check it yourself:**

```python
def greedy(coins, amount):
    out = []
    for c in sorted(coins, reverse=True):
        while amount >= c: amount -= c; out.append(c)
    return out
calls = 0
def best(coins, amount):                                 # try everything
    global calls; calls += 1
    if amount == 0: return 0
    return 1 + min(best(coins, amount - c) for c in coins if c <= amount)
us = [1, 5, 10, 25]
g = greedy(us, 41); print("US coins, 41 cents: greedy uses", len(g), "coins", g)
b = best(us, 41); print("  trying everything made", f"{calls:,}", "calls and found the same count:", b == len(g), "(greedy: 4 steps)")
odd = [1, 3, 4]; g = greedy(odd, 6)
print("coins 1, 3, 4 and amount 6: greedy", len(g), "coins", g, " best is", best(odd, 6), "coins (3 + 3)")
```

## Related
[[Dijkstra's Algorithm]] · [[Dynamic Programming]] · [[Heaps and Priority Queues]] · [[Trees]] · [[Sorting]] · [[Big-O]] · [[Entropy]] · [[Decision Tree]] · [[Graphs]] · [[Recursion]] · [[Python]] · [[Beam Search]]
- [[Backtracking]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, chapter on greedy algorithms (from memory; I did not open it for this note).
- Huffman, D. A., "A method for the construction of minimum-redundancy codes", 1952 (named from memory; I did not open it).
- Python documentation for `heapq` (docs.python.org).
- Figures: [Greedy_algorithm_36_cents.svg](https://commons.wikimedia.org/wiki/File:Greedy_algorithm_36_cents.svg) (Nandhp, public domain) and [Huffman_coding_visualisation.svg](https://commons.wikimedia.org/wiki/File:Huffman_coding_visualisation.svg) (Cmglee, CC BY-SA 4.0), on Wikimedia Commons. I viewed both before writing about them.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/algorithm-design Greedy algorithm:::Grab the best-looking option each time and never undo it. To make 36 cents, take the biggest coin that fits each time: 20, 10, 5, 1.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design List the steps of a greedy algorithm.::Pick a rule for "best right now" (the biggest coin that fits, the job that ends first), apply it, commit to it and never undo it, shrink the problem and repeat, and stop when nothing is left to decide.

#flashcards/algorithms/concept #flashcards/topic/algorithm-design What two properties show a greedy rule is safe?::The greedy-choice property (the greedy choice can always be part of some best answer) and optimal substructure (what remains is a smaller copy of the same problem, as in dynamic programming).

#flashcards/algorithms/concept #flashcards/topic/algorithm-design What do you do if you cannot show a greedy rule is safe?::Test it against a slower method that is always right (trying every possibility, or dynamic programming) on small cases. If greedy is ever worse, the rule is not safe.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the cost of a Huffman code and explain each symbol.
?
$$B=\sum_i f_i\,l_i\ =\ \sum_{\text{merges}}(w_x+w_y)$$
- $f_i$: the frequency of symbol $i$ (a count or a fraction). Common symbols should get short codes, because their length is multiplied by a large $f_i$.
- $l_i$: the length of its code, which is its depth in the tree (the left or right turns from the root). The algorithm tries to keep it small where $f_i$ is big.
- $w_x+w_y$: the combined weight of the two items merged at one step. Each merge pushes every symbol under it one level deeper, so each symbol pays its weight once per level.

#flashcards/algorithms/example #flashcards/topic/algorithm-design Frequencies 45, 13, 12, 16, 9, 5 per 100 symbols. What is $B$?::Merge the two smallest each time: $5+9=14$, $12+13=25$, $14+16=30$, $25+30=55$, $45+55=100$. Sum of merge weights $14+25+30+55+100=224$. By lengths (1, 3, 3, 3, 4, 4): $45+39+36+48+36+20=224$. 100 symbols take 224 bits against 300 for a fixed 3-bit code, a saving of about 25%.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the Kraft inequality for prefix-free codes and explain each symbol.
?
$$\sum_{i=1}^{n}2^{-l_i}\ \le\ 1$$
- $l_i$: the length of the code for symbol $i$. Every extra bit halves the share of "code space" a symbol uses.
- $2^{-l_i}$: the share of all possible bit strings that start with symbol $i$'s code. A length-1 code blocks half of everything, a length-3 code an eighth. In a prefix-free code no two codes may block the same strings.
- The sum $\le1$ means the shares fit in the whole. If it were above 1 the codes would overlap.

#flashcards/algorithms/example #flashcards/topic/algorithm-design Lengths 1, 3, 3, 3, 4, 4. What is the sum?::$0.5+3\times0.125+2\times0.0625=0.5+0.375+0.125=1$ (the code printed 1.0). The codes use all the available code space with no overlap, a valid prefix-free code with none wasted.

#flashcards/algorithms/formula #flashcards/topic/algorithm-design Write the bound on the average Huffman code length and explain each symbol.
?
$$H\ \le\ L\ <\ H+1$$
- $H$: the entropy in bits per symbol, the lower limit. The more predictable the text, the smaller $H$ and the more it can be squeezed.
- $L$: the average Huffman code length in bits per symbol, what you actually pay. It cannot go below $H$ for any prefix-free code.
- The gap of less than 1 bit per symbol is the price of using whole bits. When one symbol is very likely, $H$ is tiny but every symbol still needs at least 1 bit.

#flashcards/algorithms/example #flashcards/topic/algorithm-design Probabilities 0.45, 0.13, 0.12, 0.16, 0.09, 0.05 with $L=224/100$. Check the bound.::$H\approx2.22$ bits. $L=2.24$. $2.22\le2.24<3.22$ is true. The Huffman code is within 0.02 bits per symbol of the theoretical limit, and a fixed 3-bit code is 0.78 above it.

#flashcards/algorithms/image #flashcards/topic/algorithm-design ![[greedy-coins-36-cents.svg]]
What does this picture show?
?
Making 36 cents greedily: 36 minus 20 is 16, 16 minus 10 is 6, 6 minus 5 is 1, 1 minus 1 is 0, so the coins are 20, 10, 5 and 1. For this coin set the note's code finds greedy gives the fewest coins for every amount from 1 to 100. (From the note's caption: Nandhp, Wikimedia Commons, public domain.)

#flashcards/algorithms/image #flashcards/topic/algorithm-design ![[huffman-coding-visualisation.svg]]
What does this picture show?
?
Huffman coding of the message "A_DEAD_DAD_CEDED_A_BAD_BABE_A_BEADED_ABACA_BED" (the underscore is a space). In steps 2 to 6 the two rarest items are merged again and again, building a tree. Step 7 lists the final codes (A is 10, B is 1111, C is 1110, D is 01, E is 110, the underscore is 00), and step 8 is the whole message in bits. (From the note's caption: Cmglee, Wikimedia Commons.)

#flashcards/algorithms/cloze #flashcards/topic/algorithm-design Greedy does not always give the best answer, so the rule must be ==shown safe or tested== against a slower correct method.
