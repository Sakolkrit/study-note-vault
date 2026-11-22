---
tags: [algorithms, data-structures, trees, binary-search-tree, trie, programming, python]
status: growing
level: 4
tier: 2
---

# Trees

A **tree** is a way to arrange things so that each item has one "parent" above it and possibly several "children" below it, starting from a single item at the top called the **root**. It is a special kind of [[Graphs]] (connected, with no loops), and it is the natural shape for anything that branches: family trees, folders on your computer, the structure of a sentence, a [[Decision Tree]], and the pyramid inside a heap (see [[Heaps and Priority Queues]]).

## 🟢 1. Simple (age 12)
A family tree starts with one person at the top. Under them are their children, under each child are their children, and so on. Nobody has two parents in the picture, and there are no loops: you can never go down and end up back at the top.

A tree is useful for finding things fast. Imagine a guessing game: "I am thinking of a number from 1 to 100". You ask "Is it bigger than 50?" and throw away half the numbers at once. A special tree called a **binary search tree** does this for a whole collection: at every step you go left or right, and each step throws away about half of what is left.

**Example:** The folders on a computer are a tree. The root folder holds folders, which hold more folders and files. To find a photo, you go down one folder at a time, instead of looking at every file on the computer.

## 🔵 2. How it works
1. **Words.** The top item is the *root*. An item with children is a *parent*. An item with no children is a *leaf*. The number of steps down from the root to an item is its *depth*. The *height* of the tree is the depth of its deepest item. A tree is a graph, so the connections are called *edges*.
2. **A binary tree** lets every item have at most two children, called *left* and *right*.
3. **A binary search tree (BST)** adds one rule: for every item, everything in its *left* branch is smaller and everything in its *right* branch is bigger.
4. **To search a BST:** start at the root. If your key is smaller than the item here, go left. If bigger, go right. If equal, you found it. If you run out of tree, it is not there.
5. **To insert:** search for the key. When you fall off the tree, put the new item there as a leaf.
6. **To visit every item**, you can use a *traversal*. *Preorder* visits an item first, then its left branch, then its right branch. *Inorder* visits the left branch, then the item, then the right branch. *Postorder* visits the item last. *Level order* goes row by row, using a queue (see [[Stacks and Queues]]). On a BST, **inorder gives the keys in sorted order**.
7. **A trie** (said "try", from re-trie-val) is a different tree for words. Each step down is one letter, so words that start the same share the same first nodes. Looking up a word takes one step per letter.

![[binary-search-tree.svg]]
*A binary search tree with 9 numbers. Every number in the left branch of an item is smaller, and every number in the right branch is bigger. To find 7 you go 8 (left), 3 (right), 6 (right), 7. Image: Derrick Coetzee (Dcoetzee; the file page says the author is assumed), reworked by Booyabazooka, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Binary_search_tree.svg), public domain.*

![[trie-example.svg]]
*A trie holding the words "to", "tea", "ted", "ten", "A", "i", "in" and "inn". Each arrow adds one letter, so "tea", "ted" and "ten" share the nodes for "t" and "te". The blue numbers are values stored with some of the words, which this note does not use. Image: Booyabazooka (based on an image by Derrick Coetzee), modified by Superm401, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Trie_example.svg), public domain.*

## 🟠 3. Technical detail
- **Cost of search and insert** (see [[Big-O]]): both follow one path from the root, so the cost is the *height* of the tree. For a well-shaped tree that is about $\log_2 n$ (like [[Binary Search]]). For a badly shaped tree it can be $n$.
- **Shape matters:** the same keys give very different trees depending on the order they arrive in. In the code, 500 keys inserted in sorted order give a chain of height 499 (124,750 comparisons, exactly $n(n-1)/2$), and the same keys shuffled give a tree of height 20 (4,877 comparisons).
- **Balanced trees:** to avoid the chain, *self-balancing* trees (AVL trees, red-black trees) reshape themselves while inserting so the height stays about $\log n$. I do not cover or run them here. B-trees, a related kind with many children per node, are used in databases (from memory; not checked here).
- **Random order is usually fine:** if keys arrive in random order, the average depth is only about 1.39 $\log_2 n$ (equation 2 below). The danger is *sorted or nearly sorted* input.
- **Traversals are recursion:** each is a three-line recursive function (see [[Recursion]]). The code prints all four on the picture's tree.
- **Trie:** one step per letter, so a lookup takes time in proportion to the *word length*, not the number of words (see [[Hash Tables]] for another way to look up by key). It is good at "all words that start with…" (the code lists the words starting with "te" and "in"). It can use a lot of memory for words with little in common (equation 3 below).
- **BST against a heap and a hash table:** a BST keeps keys in sorted order (inorder), so it supports "next bigger key" and range questions. A [[Hash Tables]] is usually faster for plain lookup but has no order. A heap only gives the smallest (see [[Heaps and Priority Queues]]).
- **Trees in this vault:** a [[Decision Tree]] splits data with yes/no questions at each node. A [[Random Forest]] is many of them. The folders of this vault are a tree too.
- **Deleting from a BST** (three cases: no child, one child, two children) is not covered here, and I did not write or run it.
- **Limits:** the code counts comparisons and measures shapes on small trees (9 and 500 keys, and 100 random trees of 1,000). It does not time anything and does not run balanced trees.

The code covers four things: (1) the picture's tree and its four traversals, with a search, (2) sorted input against shuffled input, (3) average depth for random order, checked exactly on tiny cases and by simulation, (4) a trie and its node count.

```python
import random, itertools, math
from collections import deque
# 1) a binary search tree: smaller keys go left, bigger go right. Build the one in the picture.
class Node:
    def __init__(self, k): self.k, self.l, self.r = k, None, None
def insert(root, k, cnt):
    if root is None: return Node(k)
    cnt[0] += 1                                                              # one comparison per step down
    if k < root.k: root.l = insert(root.l, k, cnt)
    else: root.r = insert(root.r, k, cnt)
    return root
def build(keys):
    root, cnt = None, [0]
    for k in keys: root = insert(root, k, cnt)
    return root, cnt[0]
def height(t): return -1 if t is None else 1 + max(height(t.l), height(t.r))   # edges on the longest downward path
def size(t): return 0 if t is None else 1 + size(t.l) + size(t.r)
def depth_sum(t, d=0): return 0 if t is None else d + depth_sum(t.l, d + 1) + depth_sum(t.r, d + 1)
root, _ = build([8, 3, 10, 1, 6, 14, 4, 7, 13])
pre = lambda t: [] if t is None else [t.k] + pre(t.l) + pre(t.r)
ino = lambda t: [] if t is None else ino(t.l) + [t.k] + ino(t.r)
post = lambda t: [] if t is None else post(t.l) + post(t.r) + [t.k]
def level(t):
    out, q = [], deque([t])
    while q:
        n = q.popleft(); out.append(n.k)
        for c in (n.l, n.r):
            if c: q.append(c)
    return out
print('preorder  ', pre(root)); print('inorder   ', ino(root), '(sorted)'); print('postorder ', post(root)); print('level order', level(root))
print('nodes', size(root), '| edges', size(root) - 1, '| height', height(root), '| fewest possible height for 9 nodes: ceil(log2(9+1)) - 1 =', math.ceil(math.log2(10)) - 1)
def find(t, k):
    steps = 0
    while t is not None:
        steps += 1
        if k == t.k: return True, steps
        t = t.l if k < t.k else t.r
    return False, steps
print('search 7:', find(root, 7), '| search 13:', find(root, 13), '| search 5 (absent):', find(root, 5))
# 2) the same keys in a bad order make a chain
for n in (500,):
    chain, c1 = build(range(n)); shuf = list(range(n)); random.seed(2); random.shuffle(shuf); rnd, c2 = build(shuf)
    print('n', n, '| sorted input: height', height(chain), ', comparisons', c1, '(n(n-1)/2 =', n * (n - 1) // 2, ') | shuffled: height', height(rnd), ', comparisons', c2)
# 3) average depth of a node in a tree built from random order: exact formula 2(1 + 1/n)H_n - 4
H = lambda n: sum(1 / i for i in range(1, n + 1))
formula = lambda n: 2 * (1 + 1 / n) * H(n) - 4
for n in (3, 4):
    avg = sum(depth_sum(build(p)[0]) for p in itertools.permutations(range(n))) / math.factorial(n) / n
    print('n', n, 'exact average over all', math.factorial(n), 'orders:', round(avg, 4), '| formula:', round(formula(n), 4))
random.seed(5); n = 1000
runs = []
for _ in range(100): ks = list(range(n)); random.shuffle(ks); runs.append(depth_sum(build(ks)[0]) / n)
print('n 1000: average depth over 100 random trees', round(sum(runs) / 100, 3), '| formula', round(formula(1000), 3), '| log2(1000) =', round(math.log2(1000), 2))
# 4) a trie: one letter per step, words that share a start share nodes
class Trie:
    def __init__(self): self.kids, self.end = {}, False
def add(t, w):
    for ch in w: t = t.kids.setdefault(ch, Trie())
    t.end = True
def count(t): return 1 + sum(count(c) for c in t.kids.values())
def words_with(t, prefix):
    steps = 0
    for ch in prefix:
        if ch not in t.kids: return [], steps
        t = t.kids[ch]; steps += 1
    out = []
    def walk(n, s):
        if n.end: out.append(s)
        for ch in sorted(n.kids): walk(n.kids[ch], s + ch)
    walk(t, prefix); return out, steps
trie, words = Trie(), ['to', 'tea', 'ted', 'ten', 'A', 'i', 'in', 'inn']
for w in words: add(trie, w)
print('trie nodes', count(trie), '| 1 + total letters =', 1 + sum(map(len, words)), '| words starting with "te":', words_with(trie, 'te'), '| "in":', words_with(trie, 'in'), '| "x":', words_with(trie, 'x'))
# preorder   [8, 3, 1, 6, 4, 7, 10, 14, 13]
# inorder    [1, 3, 4, 6, 7, 8, 10, 13, 14] (sorted)
# postorder  [1, 4, 7, 6, 3, 13, 14, 10, 8]
# level order [8, 3, 10, 1, 6, 14, 4, 7, 13]
# nodes 9 | edges 8 | height 3 | fewest possible height for 9 nodes: ceil(log2(9+1)) - 1 = 3
# search 7: (True, 4) | search 13: (True, 4) | search 5 (absent): (False, 4)
# n 500 | sorted input: height 499 , comparisons 124750 (n(n-1)/2 = 124750 ) | shuffled: height 20 , comparisons 4877
# n 3 exact average over all 6 orders: 0.8889 | formula: 0.8889
# n 4 exact average over all 24 orders: 1.2083 | formula: 1.2083
# n 1000: average depth over 100 random trees 11.073 | formula 10.986 | log2(1000) = 9.97
# trie nodes 11 | 1 + total letters = 19 | words starting with "te": (['tea', 'ted', 'ten'], 2) | "in": (['in', 'inn'], 2) | "x": ([], 0)
```
Reading the output (Python 3.9, standard library only):
- **Traversals:** preorder `8 3 1 6 4 7 10 14 13`, inorder `1 3 4 6 7 8 10 13 14` (sorted), postorder `1 4 7 6 3 13 14 10 8`, level order `8 3 10 1 6 14 4 7 13`.
- **Shape:** 9 nodes, 8 edges, height 3. The smallest possible height for 9 nodes is also 3.
- **Searches:** 7 takes 4 steps, 13 takes 4, and the absent 5 takes 4 steps before running out of tree.
- **Chain against tree:** 500 sorted keys give height 499 and 124,750 comparisons. The same keys shuffled give height 20 and 4,877 comparisons.
- **Average depth:** for $n=3$ the exact average over all 6 orders is 0.8889 and for $n=4$ over all 24 orders it is 1.2083. The formula gives the same two numbers. For $n=1000$ the average over 100 random trees is 11.073 and the formula gives 10.986 (and $\log_2 1000$ is 9.97).
- **Trie:** the 8 words, with 18 letters in total, need 11 nodes (the formula's bound is 19). Words starting with "te" are `tea ted ten` (found after 2 steps), with "in" are `in inn`, and "x" gives nothing.
- **Limits:** small trees, counts only, nothing timed.

## 🔴 4. Deep / formal

**How short can a tree be?** A tree with $n$ nodes always has $n-1$ edges (every node except the root has exactly one edge up to its parent). A binary tree of height $h$ has at most $1+2+4+\dots+2^h=2^{h+1}-1$ nodes (each row can at most double). So for $n$ nodes the height is at least:

$$h\ \ge\ \lceil\log_2(n+1)\rceil-1,\qquad\text{edges}=n-1$$

**Where:**
- $n$ is the number of nodes. **Matters:** edges are one fewer than nodes, which is also why a tree is the thinnest connected graph (remove any edge and it falls into two pieces).
- $h$ is the height, counted in edges from the root to the deepest leaf. **Matters:** it is the worst-case number of steps for a search, minus one. This is the quantity we want to keep small.
- $2^{h+1}-1$ is the most nodes a binary tree of height $h$ can hold. **Matters:** it comes from filling every row, so the tree can hold *exponentially* many nodes in a small height. That is why a balanced tree with a million nodes is only about 20 high.
- $\lceil\cdot\rceil$ means "round up". **Matters:** the height is a whole number, so it cannot be less than the (fractional) value.

**Example (tiny numbers):** the picture's tree, $n=9$.
1. Edges: $9-1=8$. The code counted 8.
2. $\lceil\log_2 10\rceil-1=\lceil3.32\rceil-1=4-1=3$, so no binary tree with 9 nodes is shorter than 3.
3. The picture's tree has height 3. The code gave 3.
4. Meaning: the picture's tree is as short as a 9-node binary tree can be. Compare the chain of 500 sorted keys: height 499, against the minimum of $\lceil\log_2 501\rceil-1=8$ (from the formula; the code did not compute it).

**How deep is a typical node?** Insert $n$ distinct keys in a random order (every order equally likely) into a BST, with the root at depth 0. The average depth of a node, averaged over all orders, is:

$$D_n=\frac{2(n+1)H_n-4n}{n}=2\Big(1+\frac1n\Big)H_n-4\ \approx\ 2\ln n-2.85$$

**Where:**
- $n$ is the number of keys. **Matters:** the average depth grows like $\ln n$, so random BSTs are close to balanced.
- $H_n=1+\frac12+\dots+\frac1n$ is the $n$-th harmonic number. **Matters:** it grows slowly (about $\ln n$) and comes from this reasoning: when you insert items in random order, the key that arrives first becomes the root, and the remaining keys split into a left and a right group.
- $2(n+1)H_n-4n$ is the total of all node depths, averaged over orders. **Matters:** dividing by $n$ gives the average depth, and the cost of searching for a key that is in the tree is the depth plus one comparison.
- $2\ln n=2\ln2\cdot\log_2n\approx1.39\log_2n$. **Matters:** a random BST is only about 39% deeper than the perfect tree on average. The $-2.85$ is $2(\gamma-2)$ with $\gamma\approx0.5772$ (Euler's constant), which I worked out here and checked only as 10.97 against the exact 10.986 at $n=1000$.
- "Random order" is an assumption. **Matters:** for sorted input the average depth is $(n-1)/2$ and the formula does not apply.

I took the exact formula from memory of the standard analysis and tested it three ways: exactly for $n=3$ and $n=4$ by trying every order, and by simulation at $n=1000$. I did not prove it.

**Example (tiny numbers):** $n=3$ and $n=4$.
1. $n=3$: $H_3=1.8333$, so $D_3=2\cdot\frac43\cdot1.8333-4=0.8889$.
2. Check: of the 6 orders of 3 keys, 2 give a balanced tree (depths 0, 1, 1, total 2) and 4 give a chain (depths 0, 1, 2, total 3). The average total is $(2\cdot2+4\cdot3)/6=2.667$, and divided by 3 nodes is 0.8889. The code agreed.
3. $n=4$: the formula gives 1.2083, and the code's average over all 24 orders also gave 1.2083.
4. For $n=1000$ the formula gives 10.986, and the average over 100 random trees was 11.073. The two differ by 0.09, which fits the random variation between the 100 trees (I did not compute a confidence interval).
5. Meaning: a typical node in a random tree of 1,000 keys sits about 11 steps down, not 500. Random insertion order is almost as good as balancing.

**The size of a trie.** A trie for a set of words has a root, and then at most one node for each letter of each word, because a letter that repeats a shared start reuses an existing node:

$$\#\text{nodes}\ \le\ 1+\sum_{w}|w|$$

**Where:**
- $w$ ranges over the words and $|w|$ is the number of letters in word $w$. **Matters:** the sum is the total number of letters, which is the most new nodes the words can create.
- The $1$ is the root, the empty prefix. **Matters:** every word starts there, and it holds no letter.
- The bound is reached when no two words share a start. **Matters:** in that worst case a trie wastes memory (a node per letter, each with its own table of children). When words share many prefixes the trie is much smaller than the bound.
- A lookup of word $w$ takes $|w|$ steps, whatever the number of words. **Matters:** the cost depends on the length of the word, not on how many words are stored.

**Example (tiny numbers):** the picture's 8 words: to, tea, ted, ten, A, i, in, inn.
1. Letters: $2+3+3+3+1+1+2+3=18$. The bound is $1+18=19$.
2. The code built the trie and counted 11 nodes: the root, then t, to, te, tea, ted, ten, A, i, in, inn.
3. Meaning: sharing starts saves 8 nodes here ("tea", "ted" and "ten" share "t" and "te", and "in" and "inn" share "i" and "in"). Finding "ten" takes 3 steps, and the code found words starting with "te" after 2.

**Other points (from memory, not checked here):**
- The name "trie" is usually credited to Edward Fredkin (1960) and the word comes from "retrieval". I did not check the history.
- AVL trees (1962) were the first self-balancing BSTs, and red-black trees are widely used in standard libraries. I did not check either claim and did not run either structure.
- B-trees are used for database indexes and file systems. I did not run or check them.
- A tree can be stored in a list the way a heap is (see [[Heaps and Priority Queues]]), but a BST usually uses pointers.
- Deleting a key from a BST can unbalance it further. I did not test this.

**What came before, and how this improved on it:**
- **Before:** a sorted list (fast to search, slow to insert into) or an unsorted list (fast to add to, slow to search).
- **Its limit:** one of the two operations always costs about $n$ steps.
- **What changed:** a binary search tree keeps each smaller key to the left and each larger key to the right, so both searching and inserting follow one path from the top. Self-balancing trees such as AVL trees (Adelson-Velsky and Landis, 1962; from memory, not checked) keep that path short.
- **How much better:** in the code below, with 20,000 keys inserted in random order, a lookup took 17.9 steps on average, against 10,000 for a scan. **Honest catch:** with keys inserted already in sorted order, the plain tree became a chain and took 1,000.5 steps on average for 2,000 keys, no better than a scan. That is why balancing matters.

**Check it yourself:**

```python
import random, math
random.seed(0)
class N:
    def __init__(s, k): s.k, s.l, s.r = k, None, None
def insert(root, k):
    if root is None: return N(k)
    node = root
    while True:
        side = "l" if k < node.k else "r"
        if getattr(node, side) is None: setattr(node, side, N(k)); return root
        node = getattr(node, side)
def steps(root, k):
    c = 1
    while root.k != k: root = root.l if k < root.k else root.r; c += 1
    return c
n = 20000
for name, keys in (("random order", random.sample(range(10**6), n)), ("already sorted", list(range(n)))):
    root = None
    for k in keys[:n if name == "random order" else 2000]: root = insert(root, k)
    used = keys[:n if name == "random order" else 2000]
    avg = sum(steps(root, k) for k in used) / len(used)
    print(f"{name:15} {len(used):,} keys: {avg:,.1f} steps per lookup on average (scan would average {len(used) / 2:,.0f}; log2 n = {math.log2(len(used)):.1f})")
```

## Related
[[Graphs]] · [[Recursion]] · [[Binary Search]] · [[Stacks and Queues]] · [[Heaps and Priority Queues]] · [[Hash Tables]] · [[Big-O]] · [[Sorting]] · [[Arrays and Lists]] · [[Decision Tree]] · [[Random Forest]] · [[Python]] · [[Dijkstra's Algorithm]] · [[Greedy Algorithms]] · [[SQL]]
- [[Backtracking]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, chapters on binary search trees and elementary data structures (from memory; I did not open it for this note).
- Knuth's analysis of the average depth of random binary search trees (named from memory as the source of the standard result; I did not open it).
- Figures: [Binary_search_tree.svg](https://commons.wikimedia.org/wiki/File:Binary_search_tree.svg) (Derrick Coetzee, assumed, reworked by Booyabazooka, public domain) and [Trie_example.svg](https://commons.wikimedia.org/wiki/File:Trie_example.svg) (Booyabazooka and Superm401, public domain), on Wikimedia Commons. I viewed both before writing the captions.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/data-structures Tree:::A family tree: one root at the top, children below, and no loops. A tree is a graph, so its connections are called edges.

#flashcards/algorithms/concept #flashcards/topic/data-structures Define root, parent, leaf, depth and height.::The root is the top item. A parent is an item with children. A leaf has no children. The depth of an item is the number of steps down from the root. The height of the tree is the depth of its deepest item.

#flashcards/algorithms/reverse #flashcards/topic/data-structures Binary search tree (BST):::A binary tree (each item has at most two children, left and right) where for every item everything in its left branch is smaller and everything in its right branch is bigger.

#flashcards/algorithms/concept #flashcards/topic/data-structures How do you search and insert in a BST?::Search: start at the root, go left if your key is smaller, right if bigger, found if equal, and not there if you run out of tree. Insert: search for the key, and when you fall off the tree put the new item there as a leaf.

#flashcards/algorithms/concept #flashcards/topic/data-structures Name the four traversals.::Preorder visits the item first, then its left branch, then its right. Inorder visits left, item, right. Postorder visits the item last. Level order goes row by row.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the lower bound on the height of a binary tree and the number of edges, and explain each symbol.
?
$$h\ \ge\ \lceil\log_2(n+1)\rceil-1,\qquad\text{edges}=n-1$$
- $n$: the number of nodes. Edges are one fewer than nodes, which is why a tree is the thinnest connected graph (remove any edge and it falls into two pieces).
- $h$: the height, counted in edges from the root to the deepest leaf. It is the worst-case number of steps for a search, minus one, the quantity we want small.
- $2^{h+1}-1$: the most nodes a binary tree of height $h$ can hold. It comes from filling every row, so the tree can hold exponentially many nodes in a small height.

#flashcards/algorithms/example #flashcards/topic/data-structures The picture's tree has $n=9$ nodes. What are the edges, the minimum height, and how does a chain of 500 sorted keys compare?::Edges $9-1=8$. $\lceil\log_2 10\rceil-1=4-1=3$, so no binary tree with 9 nodes is shorter than 3, and the picture's tree has height 3 (the code gave 3). A chain of 500 sorted keys has height 499 against a minimum of $\lceil\log_2 501\rceil-1=8$ (from the formula, not computed in the code).

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the average depth of a random binary search tree and explain each symbol.
?
$$D_n=\frac{2(n+1)H_n-4n}{n}=2\Big(1+\frac1n\Big)H_n-4\ \approx\ 2\ln n-2.85$$
- $n$: the number of keys. The average depth grows like $\ln n$, so random BSTs are close to balanced.
- $H_n=1+\frac12+\dots+\frac1n$: the $n$-th harmonic number, growing about like $\ln n$. It comes from inserting in random order: the first key becomes the root and the rest split into a left and a right group.
- $2(n+1)H_n-4n$: the total of all node depths, averaged over orders. Dividing by $n$ gives the average depth, and the cost of a search is about that.

#flashcards/algorithms/example #flashcards/topic/data-structures What is $D_3$, $D_4$, and $D_{1000}$?::$n=3$: $D_3=2\cdot\frac43\cdot1.8333-4=0.8889$. Check: of 6 orders, 2 give a balanced tree (depths 0, 1, 1: total 2) and 4 give a chain (0, 1, 2: total 3), so $(2\cdot2+4\cdot3)/6=2.667$, divided by 3 nodes is 0.8889. $n=4$: 1.2083 (the code's average over all 24 orders agrees). $n=1000$: the formula gives 10.986 and the average over 100 random trees was 11.073.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the bound on the number of nodes in a trie and explain each symbol.
?
$$\#\text{nodes}\ \le\ 1+\sum_{w}|w|$$
- $w$: ranges over the words. $|w|$: the number of letters in word $w$. The sum is the total number of letters, the most new nodes the words can create.
- The $1$: the root, the empty prefix. Every word starts there and it holds no letter.
- The bound is reached when no two words share a start. Then a trie wastes memory (a node per letter, each with its own table of children). When words share many prefixes the trie is much smaller than the bound.
- A lookup of word $w$ takes about $|w|$ steps.

#flashcards/algorithms/concept #flashcards/topic/data-structures A trie holds "to", "tea", "ted", "ten", "A", "i", "in", "inn". What is the bound and the real count?::Letters $2+3+3+3+1+1+2+3=18$, so the bound is 19. The code built the trie and counted 11 nodes (the root, then t, to, te, tea, ted, ten, A, i, in, inn). Sharing starts saves 8 nodes. Finding "ten" takes 3 steps.

#flashcards/algorithms/image #flashcards/topic/data-structures ![[binary-search-tree.svg]]
What does this picture show?
?
A binary search tree with 9 numbers. Every number in the left branch of an item is smaller and every number in the right branch is bigger. To find 7 you go 8 (left), 3 (right), 6 (right), 7. (From the note's caption: Derrick Coetzee, reworked by Booyabazooka, Wikimedia Commons, public domain.)

#flashcards/algorithms/image #flashcards/topic/data-structures ![[trie-example.svg]]
What does this picture show?
?
A trie holding the words "to", "tea", "ted", "ten", "A", "i", "in" and "inn". Each arrow adds one letter, so "tea", "ted" and "ten" share the nodes for "t" and "te". The blue numbers are values stored with some of the words, which the note does not use. (From the note's caption: Booyabazooka, modified by Superm401, Wikimedia Commons, public domain.)

#flashcards/algorithms/cloze #flashcards/topic/data-structures A BST kept in sorted insertion order degenerates into a ==chain== with height $n-1$.
