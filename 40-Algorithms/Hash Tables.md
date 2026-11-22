---
tags: [algorithms, data-structures, hash-table, hashing, python, programming]
status: growing
level: 4
tier: 2
---

# Hash Tables

A hash table is a way to store pairs of (key, value) so that you can **find a value from its key in about one step**, however many pairs there are. It does this by turning the key into a number, and using that number as the address where the pair is kept. Python's `dict` and `set` are hash tables, and the speed described in [[Big-O]] for "looking up a key" comes from them.

## 🟢 1. Simple (age 12)
Imagine a wall of numbered mailboxes at school. To store a student's note, you do not search the wall. You take the student's name, run a fixed little recipe on it that turns the name into a mailbox number (say, add up the letters and take the remainder after dividing by 100), and put the note in that box. To find it again, you run the same recipe on the name and walk straight to the box. You never look at the other boxes. The recipe is called a **hash function**.

**Example:** A class has 30 students, and the wall has 100 boxes. Most students get their own box. Sometimes the recipe gives two students the same number, and they have to share a box (or one moves to the next free box). That is called a **collision**. A good recipe spreads names evenly, so sharing is rare.

## 🔵 2. How it works
1. **Make a row of buckets.** A bucket is a numbered slot. Let there be $m$ of them (say 8).
2. **Turn the key into a number** with a hash function. The same key must always give the same number.
3. **Squash it into the row** by taking the remainder after dividing by $m$. This is the key's bucket number.
4. **To store** a pair, put it in that bucket.
5. **To find** a value, compute the bucket number again and look only in that bucket.
6. **Collisions:** two different keys may land in the same bucket. There are two common fixes:
   - **Chaining:** each bucket holds a short list, and colliding keys share it. To find a key, walk the short list.
   - **Open addressing (for example, linear probing):** if the bucket is taken, try the next one, then the next, until a free one is found. To look up a key, follow the same trail.
7. **Keep it from getting crowded.** When the table is about three-quarters full, make a bigger one (twice the buckets) and put every pair into its new bucket. Rare, but it keeps lookups fast.

![[hash-table-chaining.svg]]
*A hash table that uses chaining. On the left are the keys (people's names). In the middle are the numbered buckets, and on the right are the stored entries (name and phone number). The hash function sends each key to a bucket, shown by the arrows. John Smith and Sandra Dee are sent to the same bucket (152, in red), so their entries are chained in a short list. Lisa Smith, Ted Baker and Sam Doe each get a bucket to themselves. Image: Jorge Stolfi, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Hash_table_5_0_1_1_1_1_1_LL.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Average cost is $O(1)$:** if the hash function spreads keys evenly and the table is not too full, a lookup, an insert or a delete takes a roughly constant number of steps, however many pairs are stored (see [[Big-O]]). The worst case, when all keys land in one bucket, is $O(n)$.
- **The hash function** must be *repeatable* (the same key always gives the same number) and should *spread* keys evenly. A poor one, such as "use the first letter", piles many keys into few buckets. The code uses a small text hash called djb2 (from memory, a well-known simple choice) to keep results repeatable.
- **Load factor** $\alpha=n/m$: the number of stored pairs divided by the number of buckets. It is the main measure of how crowded the table is. The costs of lookups grow with $\alpha$ (equation 2 below).
- **Chaining against probing:**
  - **Chaining** degrades gently: the average chain length equals $\alpha$, and the table still works when $\alpha>1$. It uses extra memory for the lists.
  - **Linear probing** keeps everything in one array and is friendly to the computer's cache, but its cost *explodes* as $\alpha$ approaches 1, because runs of filled slots grow and merge (the code measures it, and the figure below shows it).
- **Resizing:** when $\alpha$ passes a limit, the table doubles and every pair is placed again (the code's `grow`). One resize is costly, but because it happens rarely, each insert is still cheap *on average*. This is the amortised cost idea from [[Big-O]].
- **No order by key:** a hash table does not keep keys sorted, so you cannot ask for "the next larger key" quickly. A sorted structure such as a tree is for that (see [[Sorting]] and [[Binary Search]]).
- **Keys must be hashable, and must not change:** if a key changed after it was stored, its bucket number would change and the pair could no longer be found. This is why Python allows strings, numbers and tuples of them as dictionary keys, and refuses lists and dictionaries (the code shows the error).
- **Python's built-in hash for text changes between runs:** Python mixes a random value into the hash of text for security reasons (from memory, a feature called hash randomisation). The code shows that `hash('cat')` differs between two separate runs, but is the same within one run. So never save built-in hash values in a file and expect them to match later.
- **Python's `dict` keeps insertion order** (guaranteed from Python 3.7; from memory). That is a property of the implementation, and is separate from hashing itself.
- **Where hashing shows up in this vault:** a [[Pandas]] `merge` and `groupby` use hashing to match keys quickly (from memory; I did not check the implementation). A [[Vector Database]] uses other kinds of hashing and indexes to find *similar* items, not equal ones. [[Tokenization]] looks tokens up in a table to turn text into numbers, and a "vocabulary" is a dictionary (see [[Byte-Pair Encoding]]).
- **Hashing is not encryption:** a hash function for a table only needs to spread keys evenly. Hash functions meant for security (checksums, passwords) have stricter requirements and are a different topic, not covered here.

![[hash-table-chaining-vs-probing.png]]
*Average cache misses per lookup, against the load factor (how full the table is), for chaining and for linear probing. Linear probing is cheaper while the table is fairly empty, because everything sits close together in memory. It rises very steeply as the table gets above roughly 80% full, and passes chaining near a load factor of 0.85 to 0.9. Chaining rises only gently. Image: Derrick Coetzee, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Hash_table_average_insertion_time.png), public domain. The file page describes the vertical axis as the average number of cache misses expected, a good stand-in for real time on modern machines. I did not check how the curves were produced.*

The code does five things: (1) shows a repeatable text hash, (2) builds a small chaining table that grows, (3) measures average search steps for chaining and linear probing at three load factors and compares them with the formulas, (4) checks the chance of a collision, and (5) tests Python's own `dict` and `set`.

```python
import math, random, subprocess, sys, time
def djb2(s):                                           # a small, repeatable text hash (Python's own str hash changes between runs)
    h = 5381
    for ch in s: h = h * 33 + ord(ch)
    return h
print('djb2("cat") =', djb2('cat'), '| bucket in a table of 11:', djb2('cat') % 11, '| same call again:', djb2('cat') % 11)
# 1) a hash table with chaining, written out
class ChainTable:
    def __init__(self, m=8): self.m = m; self.b = [[] for _ in range(m)]; self.n = 0
    def put(self, k, v):
        chain = self.b[djb2(k) % self.m]
        for i, (kk, _) in enumerate(chain):
            if kk == k: chain[i] = (k, v); return
        chain.append((k, v)); self.n += 1
        if self.n > 0.75 * self.m: self.grow()
    def get(self, k):
        for kk, v in self.b[djb2(k) % self.m]:
            if kk == k: return v
        raise KeyError(k)
    def grow(self):                                    # double the buckets and re-place every item
        old = [kv for chain in self.b for kv in chain]; self.m *= 2; self.b = [[] for _ in range(self.m)]
        for k, v in old: self.b[djb2(k) % self.m].append((k, v))
t = ChainTable()
for i, w in enumerate(['cat', 'dog', 'emu', 'fox', 'gnu', 'hen', 'ibis', 'jay']): t.put(w, i)
print('items', t.n, '| buckets now', t.m, '| get fox:', t.get('fox'), '| longest chain:', max(len(c) for c in t.b))
# 2) average steps against how full the table is (load factor a = n/m); keys are random strings
def keys(n, rng): return [''.join(rng.choice('abcdefghijklmnop') for _ in range(12)) for _ in range(n)]
def chain_cost(n, m, rng):
    ks = keys(n, rng); b = [[] for _ in range(m)]
    for k in ks: b[djb2(k) % m].append(k)
    succ = sum(b[djb2(k) % m].index(k) + 1 for k in ks) / n                       # items looked at to find a key that is there
    miss = sum(len(b[djb2(k) % m]) for k in keys(n, rng)) / n                     # items looked at for a key that is not there
    return succ, miss
def probe_cost(n, m, rng):
    ks = keys(n, rng); slots = [None] * m
    def find(k):
        i, c = djb2(k) % m, 1
        while slots[i] is not None and slots[i] != k: i = (i + 1) % m; c += 1
        return i, c
    for k in ks: slots[find(k)[0]] = k
    return sum(find(k)[1] for k in ks) / n, sum(find(k)[1] for k in keys(n, rng)) / n
rng = random.Random(1); m = 2003
print('load | chaining found/missing (theory 1+a/2, a) | linear probing found/missing (theory .5(1+1/(1-a)), .5(1+1/(1-a)^2))')
for a in (0.5, 0.8, 0.9):
    n = int(a * m); cs, cm = chain_cost(n, m, rng); ps, pm = probe_cost(n, m, rng)
    print(a, '|', round(cs, 2), round(cm, 2), f'({1 + a / 2:.2f}, {a:.2f})', '|', round(ps, 2), round(pm, 2), f'({0.5 * (1 + 1 / (1 - a)):.2f}, {0.5 * (1 + 1 / (1 - a) ** 2):.2f})')
# 3) collisions come sooner than you expect (the birthday effect): chance that n keys in m buckets have no collision
def p_none(n, m): return math.prod(1 - i / m for i in range(n))
print('m=365, n=23: P(no collision) =', round(p_none(23, 365), 4), '| approx exp(-n(n-1)/2m) =', round(math.exp(-23 * 22 / (2 * 365)), 4))
rs = random.Random(0); trials = 20000; hits = sum(len({rs.randrange(365) for _ in range(23)}) < 23 for _ in range(trials))
print('simulated P(at least one collision):', round(hits / trials, 3), '| m=1000000, n=1000: expected colliding pairs', round(1000 * 999 / 2 / 10 ** 6, 3))
# 4) Python's own dict and set: lookups by hash, and why keys must not change
n = 100000; L = list(range(n)); S = set(L)
t0 = time.perf_counter(); [n - 1 in L for _ in range(20)]; tl = (time.perf_counter() - t0) / 20
t0 = time.perf_counter(); [n - 1 in S for _ in range(20)]; ts = (time.perf_counter() - t0) / 20
print('looking up the last of 100000 items: list scan is about', round(tl / ts), 'times slower than the set (timing varies with the machine)')
for bad in ([1, 2], {'a': 1}):
    try: {bad: 1}
    except TypeError as e: print('unhashable:', type(bad).__name__, '->', e)
h = [subprocess.run([sys.executable, '-c', "print(hash('cat'))"], capture_output=True, text=True).stdout.strip() for _ in range(2)]
print('built-in hash of "cat" in two separate runs differs:', h[0] != h[1], '| but within one run hash("cat") == hash("cat"):', hash('cat') == hash('cat'))
# djb2("cat") = 193488125 | bucket in a table of 11: 6 | same call again: 6
# items 8 | buckets now 16 | get fox: 3 | longest chain: 2
# load | chaining found/missing (theory 1+a/2, a) | linear probing found/missing (theory .5(1+1/(1-a)), .5(1+1/(1-a)^2))
# 0.5 | 1.23 0.48 (1.25, 0.50) | 1.51 2.61 (1.50, 2.50)
# 0.8 | 1.38 0.81 (1.40, 0.80) | 3.0 11.49 (3.00, 13.00)
# 0.9 | 1.47 0.91 (1.45, 0.90) | 5.25 34.65 (5.50, 50.50)
# m=365, n=23: P(no collision) = 0.4927 | approx exp(-n(n-1)/2m) = 0.5
# simulated P(at least one collision): 0.509 | m=1000000, n=1000: expected colliding pairs 0.499
# looking up the last of 100000 items: list scan is about 1517 times slower than the set (timing varies with the machine)   <- 2374 on an earlier run
# unhashable: list -> unhashable type: 'list'
# unhashable: dict -> unhashable type: 'dict'
# built-in hash of "cat" in two separate runs differs: True | but within one run hash("cat") == hash("cat"): True
```
Reading the output (Python 3.9, standard library only):
- **Repeatable:** the same key gives the same bucket every time (`cat` lands in bucket 6 of 11).
- **The table grows:** 8 pairs in a table that began with 8 buckets triggered one doubling to 16 buckets. A lookup for `fox` finds its value 3. The longest chain has 2 items, so a collision did happen.
- **Chaining costs, found and missing:** the measured steps follow the simple formulas closely. At load 0.5, a found key took 1.23 steps and a missing key looked at 0.48 items (formulas 1.25 and 0.50). At load 0.9, 1.47 and 0.91 (formulas 1.45 and 0.90). Even at 90% full, chaining stays cheap.
- **Linear probing costs:** it is close to the formulas at load 0.5 (1.51 and 2.61 against 1.50 and 2.50). At 0.8 it needs 3.0 steps for a found key and 11.49 for a missing one (formulas 3.00 and 13.00). At 0.9 a missing key needed 34.65 steps, against a formula value of 50.5. The shape is right (it explodes), but the formula for missing keys is for the long-run average of ideal random hashing, and at this table size (2,003 slots) the measured values are noisy and fell below it at the highest loads. I used one random seed and did not repeat the runs, so I cannot say how big the scatter is.
- **Birthday effect:** with 23 keys in 365 buckets, the chance of *no* collision is 0.4927 by the exact product and 0.5 by the approximation. In 20,000 simulated trials, 50.9% had at least one collision, which agrees with $1-0.4927=0.507$. For 1,000 keys in a million buckets, the expected number of colliding pairs is about 0.5, so even that big table has a real chance of at least one collision.
- **Speed:** looking up the last item of 100,000 numbers was roughly a thousand or more times faster in a set than in a list on my machine (1,517 times on the run in the comment above, 2,374 times on an earlier run). The ratio depends on the machine and changes from run to run, so only the size of the gap matters.
- **Python's rules:** lists and dictionaries cannot be keys. The built-in text hash differs between two separate runs of Python.

## 🔴 4. Deep / formal

**Which bucket.** A hash function $h$ turns a key $k$ into a (large) whole number, and the bucket among $m$ is

$$i=h(k)\bmod m$$

**Where:**
- $k$ is the key (for example, a text such as "cat"). **Matters:** only keys that are equal must give the same number. Keys that are close or similar should be spread widely.
- $h(k)$ is the hash value, a large whole number. **Matters:** it must be repeatable, since otherwise a stored pair could never be found again.
- $m$ is the number of buckets. **Matters:** it sets how many places there are. Prime sizes, or sizes chosen with care, are sometimes preferred to avoid patterns in the hash values (from memory, not tested here).
- $\bmod$ is the remainder after division. **Matters:** it folds a huge number into a small address between $0$ and $m-1$. Different keys can land on the same address, which is a collision.
- $i$ is the bucket number. **Matters:** it is where the pair is stored and later looked for.

**Example (tiny numbers):** The code's hash $h=\text{djb2}$ on three-letter words, with $m=8$ buckets.
1. $h(\text{cat})=193{,}488{,}125$, and $193{,}488{,}125\bmod8=5$, so cat goes to bucket 5.
2. $h(\text{dog})=193{,}489{,}663$, giving bucket 7. $h(\text{gnu})$ also gives bucket 7. That is a collision: dog and gnu share a bucket.
3. $h(\text{emu})$ and $h(\text{ibis})$ both give bucket 4, another collision.
4. After the table doubles to $m=16$: cat is at 13, dog and gnu are both at 15, emu and ibis are both at 12. These pairs still collide.
5. Meaning: doubling $m$ separates *some* keys but not these, which happen to agree even modulo 16. A resize helps on average and does not guarantee a collision-free table.

**How many steps a search takes.** Let $\alpha=n/m$ be the load factor. With a good hash function and random-like keys, the average number of items or slots looked at is:

$$\text{chaining: } 1+\tfrac{\alpha}{2}\ \text{(key present)},\ \ \alpha\ \text{(absent)};\qquad \text{linear probing: } \tfrac12\!\Big(1+\tfrac1{1-\alpha}\Big)\ \text{(present)},\ \ \tfrac12\!\Big(1+\tfrac1{(1-\alpha)^2}\Big)\ \text{(absent)}$$

**Where:**
- $\alpha=n/m$ is stored pairs divided by buckets. **Matters:** it is the single number that tells you how crowded the table is. Chaining works for any $\alpha$, while probing needs $\alpha<1$ (there must be a free slot).
- For chaining, an average chain holds $\alpha$ items. **Matters:** looking for a key that is absent means reading a whole chain ($\alpha$ items). For a present key you read, on average, about half a chain plus the key itself ($1+\alpha/2$).
- For probing, the $1/(1-\alpha)$ terms. **Matters:** as $\alpha\to1$ they blow up, since long runs of occupied slots form and any key hashing into a run has to walk its whole length. The absent-key cost has a *square*, which is why it grows even faster.
- These formulas are for random-like hashing and large tables (the probing formulas are a classical result of Knuth, from memory). **Matters:** with a poor hash function or a small table, real counts can differ, as the code's 34.65 against 50.5 shows.
- The count is of items or slots examined, not of seconds. **Matters:** the real time depends on the memory cache, as the figure's discussion says.

**Example (tiny numbers):** Table load $\alpha=0.5$ and $\alpha=0.9$.
1. Chaining, present: $1+0.25=1.25$ at $\alpha=0.5$ and $1+0.45=1.45$ at $\alpha=0.9$.
2. Chaining, absent: $0.5$ and $0.9$.
3. Probing, present: $\tfrac12(1+\tfrac1{0.5})=1.5$ at $\alpha=0.5$ and $\tfrac12(1+\tfrac1{0.1})=5.5$ at $\alpha=0.9$.
4. Probing, absent: $\tfrac12(1+\tfrac1{0.25})=2.5$ at $\alpha=0.5$ and $\tfrac12(1+\tfrac1{0.01})=50.5$ at $\alpha=0.9$.
5. The code measured 1.23, 0.48, 1.51, 2.61 at $\alpha=0.5$ and 1.47, 0.91, 5.25, 34.65 at $\alpha=0.9$, close to these except for the last one.
6. Meaning: going from half full to 90% full barely changes chaining (1.25 to 1.45 steps) but makes failed lookups in linear probing about 20 times more expensive in theory (2.5 to 50.5). This is why implementations resize at around 70 to 80% full.

**Collisions come early.** If $n$ keys are placed independently and evenly into $m$ buckets, the chance that **no two** share a bucket is

$$P(\text{no collision})=\prod_{i=0}^{n-1}\Big(1-\frac im\Big)\approx e^{-\,n(n-1)/(2m)}$$

**Where:**
- $n$ is the number of keys and $m$ the number of buckets. **Matters:** a collision is likely much sooner than $n\approx m$. It becomes likely when $n$ is only around $\sqrt m$.
- The product multiplies the chance that key $i+1$ avoids the $i$ buckets already used, which is $1-i/m$. **Matters:** each new key has a slightly smaller chance of landing on an empty bucket than the one before.
- $n(n-1)/2$ is the number of pairs of keys, each of which could collide with chance $1/m$. **Matters:** pairs grow with the *square* of $n$, which is why collisions are so common. This is the "birthday problem".
- The $\approx$ is accurate when $n$ is small compared with $m$. **Matters:** it is a convenient estimate, not exact.

**Example (tiny numbers):** $m=365$ buckets (days of the year) and $n=23$ keys.
1. Number of pairs: $23\cdot22/2=253$. Each collides with chance $1/365$, so about $253/365=0.69$ colliding pairs are expected.
2. Exact: $\prod_{i=0}^{22}(1-i/365)=0.4927$. Approximation: $e^{-253/365}=e^{-0.693}=0.5000$.
3. A simulation of 20,000 trials gave a collision in 50.9% of them, which agrees with $1-0.4927=0.507$ within the noise.
4. For a big table, $m=10^6$ and $n=1000$: $1000\cdot999/2/10^6=0.499$ expected colliding pairs.
5. Meaning: with only 23 keys in 365 buckets, a collision is as likely as not, and even a million buckets for 1,000 keys has a good chance of one collision. Collisions are not a rare accident. A hash table must handle them, which is why chaining and probing exist.

**Other points (from memory, not checked here):**
- The first uses of hashing for storage date from the 1950s (credited to people at IBM). I did not check the history for this note.
- Techniques exist to make lookups worst-case constant time (for example cuckoo hashing and perfect hashing). I have not covered or tested them.
- Real hash tables, including Python's, use more careful designs than the ones here (for example open addressing with a different probing pattern). I have not looked at how CPython does it.
- The djb2 hash is a simple, well-known text hash. I did not test how evenly it spreads real-world keys. For the random strings in the code, it behaved close to the formulas.

**What came before, and how this improved on it:**
- **Before:** keep items in a list and scan it, or keep a sorted list and use [[Binary Search]].
- **Its limit:** the scan grows with the list, and a sorted list is costly to keep sorted when items are added.
- **What changed:** turn the key into a number with a hash function and jump straight to that slot. Hans Peter Luhn is usually credited with the first hash table idea at IBM in 1953 (from memory, not checked). Items that land in the same slot share a short chain.
- **How much better:** in the code below, with 20,000 keys, a list scan needed 250 comparisons on average over the first 500 keys (up to 20,000), and the hash table needed 1.33 on average and 6 at most. Python's built-in `hash` of an integer is the number itself, so the spread here is easy; real keys need a good hash function.

**Check it yourself:**

```python
import random
random.seed(0)
keys = random.sample(range(10**9), 20000)
class Table:
    def __init__(s, size): s.b = [[] for _ in range(size)]
    def put(s, k): s.b[hash(k) % len(s.b)].append(k)
    def get(s, k):
        probes = 0
        for x in s.b[hash(k) % len(s.b)]:
            probes += 1
            if x == k: break
        return probes
t = Table(30011)
for k in keys: t.put(k)
lst = keys[:]
hp = [t.get(k) for k in keys]; lp = [lst.index(k) + 1 for k in keys[:500]]
print("20,000 keys, looking each one up:")
print(f"  plain list scan: {sum(lp) / len(lp):,.0f} comparisons on average (first 500 keys), up to {len(keys):,}")
print(f"  hash table:      {sum(hp) / len(hp):.2f} comparisons on average, {max(hp)} at most")
```

## Related
[[Big-O]] · [[Python]] · [[Pandas]] · [[NumPy]] · [[Vector Database]] · [[Tokenization]] · [[Byte-Pair Encoding]] · [[Probability]] · [[Sorting]] · [[Binary Search]] · [[Dynamic Programming]] · [[Arrays and Lists]] · [[Graphs]] · [[Trees]] · [[Git]] · [[SQL]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, the chapter on hash tables (from memory; I did not open it for this note).
- Knuth, *The Art of Computer Programming*, volume 3, on searching and hashing, for the linear-probing formulas (from memory; I did not open it).
- Python documentation on `dict`, `set` and `hash()` (docs.python.org).
- Figures: [Hash_table_5_0_1_1_1_1_1_LL.svg](https://commons.wikimedia.org/wiki/File:Hash_table_5_0_1_1_1_1_1_LL.svg) (Jorge Stolfi, CC BY-SA 3.0) and [Hash_table_average_insertion_time.png](https://commons.wikimedia.org/wiki/File:Hash_table_average_insertion_time.png) (Derrick Coetzee, public domain), both on Wikimedia Commons. I viewed both before writing the captions.
- The code and every Example number were run for this note (Python 3.9, standard library only).

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/data-structures Hash table:::A wall of numbered mailboxes. A fixed recipe (the hash function) turns a key into a box number, so you store and find things without searching.

#flashcards/algorithms/concept #flashcards/topic/data-structures List the steps to store and find a pair in a hash table.::Make a row of $m$ buckets, turn the key into a number with a hash function (the same key always gives the same number), squash it into the row by taking the remainder after dividing by $m$, store the pair in that bucket, and to find it compute the bucket again and look only there.

#flashcards/algorithms/concept #flashcards/topic/data-structures What is a collision and what are the two common fixes?::Two different keys land in the same bucket. Chaining: each bucket holds a short list and colliding keys share it. Open addressing (for example linear probing): if the bucket is taken, try the next one, then the next, until a free one is found, and look up along the same trail.

#flashcards/algorithms/concept #flashcards/topic/data-structures When should a hash table grow?::When it is about three-quarters full, make a bigger one (twice the buckets) and put every pair into its new bucket. Rare, but it keeps lookups fast.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the bucket number of a key and explain each symbol.
?
$$i=h(k)\bmod m$$
- $k$: the key (for example the text "cat"). Only equal keys must give the same number. Similar keys should be spread widely.
- $h(k)$: the hash value, a large whole number. It must be repeatable, otherwise a stored pair could never be found again.
- $m$: the number of buckets. Prime sizes, or sizes chosen with care, are sometimes preferred to avoid patterns (from memory, not tested in the note).
- $\bmod$: the remainder after dividing, which squashes the big number into $0$ to $m-1$.

#flashcards/algorithms/example #flashcards/topic/data-structures With the note's djb2 hash and $m=8$, which buckets do cat, dog, gnu, emu and ibis land in?::cat 193,488,125 gives bucket 5. dog 193,489,663 gives 7, and gnu also gives 7 (a collision). emu and ibis both give 4 (another collision). After doubling to $m=16$: cat 13, dog and gnu both 15, emu and ibis both 12. These pairs still collide.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the expected lookup cost for chaining and for linear probing and explain each symbol.
?
$$\text{chaining: } 1+\tfrac{\alpha}{2}\ \text{(key present)},\ \ \alpha\ \text{(absent)};\qquad \text{linear probing: } \tfrac12\!\Big(1+\tfrac1{1-\alpha}\Big)\ \text{(present)},\ \ \tfrac12\!\Big(1+\tfrac1{(1-\alpha)^2}\Big)\ \text{(absent)}$$
- $\alpha=n/m$: stored pairs divided by buckets, the one number that tells how crowded the table is. Chaining works for any $\alpha$, while probing needs $\alpha<1$ (there must be a free slot).
- Chaining: an average chain holds $\alpha$ items. An absent key means reading a whole chain ($\alpha$ items). A present key costs about half a chain plus the key itself ($1+\alpha/2$).
- Probing: the $1/(1-\alpha)$ terms blow up as $\alpha\to1$.

#flashcards/algorithms/example #flashcards/topic/data-structures What are the four costs at $\alpha=0.5$ and $\alpha=0.9$?::Chaining present: 1.25 and 1.45. Chaining absent: 0.5 and 0.9. Probing present: 1.5 and 5.5. Probing absent: 2.5 and 50.5. The code measured 1.23, 0.48, 1.51, 2.61 at $\alpha=0.5$ and 1.47, 0.91, 5.25, 34.65 at $\alpha=0.9$, close to the formulas.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the chance that $n$ keys have no collision in $m$ buckets and explain each symbol.
?
$$P(\text{no collision})=\prod_{i=0}^{n-1}\Big(1-\frac im\Big)\approx e^{-\,n(n-1)/(2m)}$$
- $n$: the number of keys. $m$: the number of buckets. A collision becomes likely when $n$ is only around $\sqrt m$, far sooner than $n\approx m$.
- The product multiplies the chance that key $i+1$ avoids the $i$ buckets already used, $1-i/m$. Each new key has a slightly smaller chance of landing on an empty bucket.
- $n(n-1)/2$: the number of pairs of keys, each of which could collide with chance $1/m$. Pairs grow with the square of $n$.

#flashcards/algorithms/example #flashcards/topic/data-structures $m=365$ buckets and $n=23$ keys. What is the chance of no collision?::Pairs $23\cdot22/2=253$, about $253/365=0.69$ colliding pairs expected. Exact $0.4927$. Approximation $e^{-0.693}=0.5000$. A simulation of 20,000 trials gave a collision in 50.9%, consistent with $1-0.4927=0.507$. For $m=10^6$ and $n=1000$ about 0.499 colliding pairs are expected.

#flashcards/algorithms/image #flashcards/topic/data-structures ![[hash-table-chaining.svg]]
What does this picture show?
?
A hash table that uses chaining. On the left are the keys (people's names), in the middle the numbered buckets, on the right the stored entries (name and phone number). The hash function sends each key to a bucket. John Smith and Sandra Dee go to the same bucket (152, in red), so their entries are chained in a short list. Lisa Smith, Ted Baker and Sam Doe each get a bucket to themselves. (From the note's caption: Jorge Stolfi, Wikimedia Commons.)

#flashcards/algorithms/image #flashcards/topic/data-structures ![[hash-table-chaining-vs-probing.png]]
What does this picture show?
?
Average cache misses per lookup against the load factor for chaining and linear probing. Linear probing is cheaper while the table is fairly empty, because everything sits close together in memory. It rises very steeply above roughly 80% full and passes chaining near a load factor of 0.85 to 0.9. Chaining rises only gently. (From the note's caption: Derrick Coetzee, Wikimedia Commons, public domain.)
