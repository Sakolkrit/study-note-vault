---
tags: [algorithms, data-structures, graphs, bfs, dfs, programming, python, math]
status: growing
level: 4
tier: 2
---

# Graphs

A **graph** is a set of **points** (called *vertices* or *nodes*) joined by **lines** (called *edges*). It is the simplest way to describe *things and the connections between them*: towns and roads, people and friendships, web pages and links, or computers in a network. Searching a graph uses a [[Stacks and Queues]] structure to remember where to go next, and graphs are stored with the [[Arrays and Lists]] and [[Hash Tables]] you have already met. The word has nothing to do with a chart or a plot.

## 🟢 1. Simple (age 12)
Draw some dots on paper. Draw a line between two dots whenever those two things are linked. If the dots are children and the lines mean "are friends", you have a picture of who knows whom. That picture is a graph.

With it you can ask useful questions. "Can a message get from Ana to Ben by passing it between friends?" "What is the fewest number of passes?" "Is everyone in one big group, or are there separate groups?"

**Example:** A map of a school with dots for rooms and lines for corridors. To find the shortest way from the classroom to the library, you look for the route that goes through the fewest corridors.

## 🔵 2. How it works
1. **Describe the graph.** List the vertices and the edges. Edges can have a direction (a one-way street) or not (a two-way street), and they can carry a number (the length of the road, called a *weight*).
2. **Store it.** The usual way is an *adjacency list*: for every vertex, a list of the vertices it touches. The other way is an *adjacency matrix*: a square table with a 1 in row $i$, column $j$ if $i$ and $j$ are linked, and a 0 if not.
3. **Explore it with a to-do list.** Start at one vertex. Put its neighbours on a to-do list. Take one off, visit it, and add *its* neighbours (but never a vertex you have already seen).
4. **The choice of to-do list changes everything.** If you take the *oldest* item first (a queue), you spread out evenly, like ripples in a pond: this is **breadth-first search**, and it finds the route with the fewest edges. If you take the *newest* item first (a stack), you go deep along one path before backing up: this is **depth-first search**.
5. **Count the pieces.** If a search from one vertex does not reach everything, the graph is in separate pieces (*connected components*). Starting a new search at each unvisited vertex counts them.

![[bfs-algorithm.gif]]
*Breadth-first search on a graph of 6 vertices, starting at vertex 6. White vertices are not yet seen, grey ones are on the to-do list, black ones are finished. The search spreads out in rings: 6, then 4, then 3 and 5, then 2, then 1. This is an animation: some viewers show only the first frame. I viewed 7 of its 13 frames, and the code below uses this same graph. Image: Aexpedito, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Breadth-first_search_Algorithm.gif), CC BY-SA 3.0.*

![[konigsberg-bridges-map.png]]
*A map of the city of Königsberg with its seven bridges highlighted in pink. The question "can you take a walk that crosses every bridge exactly once?" led to the first result in graph theory (in the code and in section 4). Image: Twotwos, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bridges_of_Konigsberg.png), CC BY-SA 4.0. The map itself is an old engraving.*

## 🟠 3. Technical detail
- **Words:** a *path* is a walk along edges. A *cycle* is a path that returns to its start. The *degree* of a vertex is the number of edges touching it. A graph is *connected* if there is a path between any two vertices. A *tree* is a connected graph with no cycles (see [[Recursion]] for the recursive trees you may have met).
- **Adjacency list or matrix** (see [[Big-O]]): with $V$ vertices and $E$ edges, the list takes about $V+E$ space and the matrix takes $V^2$. For the 6-vertex picture, the list has 14 entries (each edge stored at both ends) and the matrix 36 cells. For a big, sparse graph (such as a road map), the list is much smaller. The matrix lets you ask "is there an edge between $i$ and $j$?" in $O(1)$.
- **Breadth-first search (BFS):** use a queue (`collections.deque`). The work is about $V+2E$ (equation 2 below). The code counts 6 vertices and 14 edge looks, making 20. BFS gives the **fewest-edges** path from the start to every other vertex. The code checks this a second way, with powers of the matrix (equation 3).
- **Depth-first search (DFS):** use recursion or a stack (see [[Recursion]]). It has the same $V+2E$ work. It is the tool for finding cycles and connected pieces, and it does **not** find shortest paths. In the code's small maze, BFS finds a 6-step route (confirmed by checking every route), while the first route DFS happens to find takes 10 steps.
- **Weighted edges:** when edges have different lengths, BFS no longer gives the shortest route. The standard fix is Dijkstra's algorithm, which uses a priority queue (see [[Dijkstra's Algorithm]], where I ran it).
- **Graphs in the vault:** a [[Neural Network]] is a graph of layers, [[Backpropagation]] walks the computation graph backwards (see [[PyTorch]]), a [[Decision Tree]] is a tree, and a [[Vector Database]] such as the HNSW index in that note searches a graph of nearby points. The Markov decision process in [[Q-Learning]] is a graph of states with edges for actions.
- **Königsberg:** the code treats the four land masses as vertices and the seven bridges as edges (some pairs have two bridges between them). The degrees are 5, 3, 3 and 3: all four are odd, and the search confirms there is no walk crossing every bridge once. After removing one bridge, only two degrees are odd, and a walk exists.
- **Limits:** the code runs tiny graphs and one small maze. It does not time anything, does not use directed or weighted graphs, and does not test huge graphs.

The code covers seven things: (1) the picture's graph as a list and a matrix, with the degree check, (2) BFS with its work counted, (3) the same distances from matrix powers, (4) counting walks with matrix powers, (5) DFS two ways and counting pieces, (6) Königsberg, (7) BFS against DFS in a maze.

```python
from collections import deque
import itertools
# 1) the graph in the picture: 6 vertices, 7 edges. Adjacency list = for each vertex, who it touches
edges = [(6, 4), (4, 5), (4, 3), (5, 1), (5, 2), (1, 2), (3, 2)]
adj = {v: [] for v in range(1, 7)}
for a, b in edges: adj[a].append(b); adj[b].append(a)
for v in adj: adj[v].sort()
deg = {v: len(adj[v]) for v in adj}
print('adjacency list:', adj)
print('degrees:', deg, '| sum of degrees', sum(deg.values()), '| 2 x edges', 2 * len(edges))
M = [[1 if j in adj[i] else 0 for j in range(1, 7)] for i in range(1, 7)]               # adjacency matrix: a 6 x 6 table of 0/1
print('matrix cells', 6 * 6, '| list entries', sum(deg.values()), '| matrix symmetric:', all(M[i][j] == M[j][i] for i in range(6) for j in range(6)))
# 2) breadth-first search: a queue; counts the work
def bfs(start):
    dist, order, q, looks = {start: 0}, [], deque([start]), 0
    while q:
        v = q.popleft(); order.append(v)
        for w in adj[v]:
            looks += 1                                                       # one look at one adjacency entry
            if w not in dist: dist[w] = dist[v] + 1; q.append(w)
    return order, dist, looks
order, dist, looks = bfs(6)
print('BFS from 6 visits', order, '| distances', dist, '| vertices', len(order), '+ edge looks', looks, '= work', len(order) + looks, '= V + 2E =', 6 + 2 * len(edges))
# 3) check the distances a second way: the smallest k for which the k-th power of the matrix has a nonzero entry
def matmul(A, B): return [[sum(A[i][k] * B[k][j] for k in range(6)) for j in range(6)] for i in range(6)]
P, first = M, {6: 0}
for k in range(1, 6):
    for j in range(1, 7):
        if j not in first and P[5][j - 1] > 0: first[j] = k                  # row 5 is vertex 6
    P = matmul(P, M)
print('shortest distances from vertex 6 by matrix powers:', dict(sorted(first.items())), '| equal to BFS:', first == dist)
# 4) powers of the matrix count walks (paths that may revisit)
M2 = matmul(M, M)
def walks(i, j, k):
    return sum(1 for mid in itertools.product(range(1, 7), repeat=k - 1) if all(b in adj[a] for a, b in zip((i,) + mid, mid + (j,))))
print('A^2 entry (3 to 5):', M2[2][4], '| brute-force count of 2-step walks 3 to 5:', walks(3, 5, 2), '| A^2 entry (4 to 4):', M2[3][3], '= degree of 4:', deg[4])
# 5) depth-first search two ways, and counting connected pieces
def dfs_rec(v, seen):
    seen.append(v)
    for w in adj[v]:
        if w not in seen: dfs_rec(w, seen)
    return seen
def dfs_stack(start):
    seen, st = [], [start]
    while st:
        v = st.pop()
        if v in seen: continue
        seen.append(v); st.extend(reversed(adj[v]))
    return seen
print('DFS from 6, recursive:', dfs_rec(6, []), '| with a stack:', dfs_stack(6))
def pieces(g):
    seen, count = set(), 0
    for v in g:
        if v not in seen: count += 1; stack = [v]; seen.add(v)
        else: continue
        while stack:
            for w in g[stack.pop()]:
                if w not in seen: seen.add(w); stack.append(w)
    return count
two = {1: [2], 2: [1], 3: [4], 4: [3, 5], 5: [4], 6: []}
print('connected pieces: picture graph', pieces(adj), '| graph with three separate parts', pieces(two))
# 6) Koenigsberg: land masses A (island), B, C (the two banks), D (east). 7 bridges. Can a walk cross each bridge exactly once?
def trail_exists(bridges):
    def go(at, used):
        if len(used) == len(bridges): return True
        for i, (x, y) in enumerate(bridges):
            if i not in used and at in (x, y) and go(y if at == x else x, used | {i}): return True
        return False
    return any(go(s, frozenset()) for s in 'ABCD')
K = [('A', 'B'), ('A', 'B'), ('A', 'C'), ('A', 'C'), ('A', 'D'), ('B', 'D'), ('C', 'D')]
dK = {c: sum(c in e for e in K) for c in 'ABCD'}
print('bridge counts at each land mass:', dK, '| odd ones:', sum(d % 2 for d in dK.values()), '| walk using every bridge once exists:', trail_exists(K))
K2 = K[1:]; dK2 = {c: sum(c in e for e in K2) for c in 'ABCD'}
print('after removing one A-B bridge:', dK2, '| odd ones:', sum(d % 2 for d in dK2.values()), '| walk exists:', trail_exists(K2))
# 7) shortest path in a maze: BFS finds it; the first path a depth-first search finds can be longer
maze = ['S....', '.#.#.', '....G', '.#.#.', '.....']
cells = {(r, c) for r in range(5) for c in range(5) if maze[r][c] != '#'}
S = next((r, c) for r in range(5) for c in range(5) if maze[r][c] == 'S'); G = next((r, c) for r in range(5) for c in range(5) if maze[r][c] == 'G')
def nbrs(p): return [q for q in ((p[0] + 1, p[1]), (p[0], p[1] + 1), (p[0] - 1, p[1]), (p[0], p[1] - 1)) if q in cells]
d, q = {S: 0}, deque([S])
while q:
    p = q.popleft()
    for n in nbrs(p):
        if n not in d: d[n] = d[p] + 1; q.append(n)
def dfs_first(p, seen):
    if p == G: return [p]
    seen.add(p)
    for n in nbrs(p):
        if n not in seen:
            r = dfs_first(n, seen)
            if r: return [p] + r
best = [10 ** 9]
def all_paths(p, seen):
    if p == G: best[0] = min(best[0], len(seen)); return
    for n in nbrs(p):
        if n not in seen: all_paths(n, seen | {n})
all_paths(S, {S})
print('maze: BFS shortest', d[G], 'steps | brute force over every simple path:', best[0] - 1, '| first path found by DFS:', len(dfs_first(S, set())) - 1, 'steps')
# adjacency list: {1: [2, 5], 2: [1, 3, 5], 3: [2, 4], 4: [3, 5, 6], 5: [1, 2, 4], 6: [4]}
# degrees: {1: 2, 2: 3, 3: 2, 4: 3, 5: 3, 6: 1} | sum of degrees 14 | 2 x edges 14
# matrix cells 36 | list entries 14 | matrix symmetric: True
# BFS from 6 visits [6, 4, 3, 5, 2, 1] | distances {6: 0, 4: 1, 3: 2, 5: 2, 2: 3, 1: 3} | vertices 6 + edge looks 14 = work 20 = V + 2E = 20
# shortest distances from vertex 6 by matrix powers: {1: 3, 2: 3, 3: 2, 4: 1, 5: 2, 6: 0} | equal to BFS: True
# A^2 entry (3 to 5): 2 | brute-force count of 2-step walks 3 to 5: 2 | A^2 entry (4 to 4): 3 = degree of 4: 3
# DFS from 6, recursive: [6, 4, 3, 2, 1, 5] | with a stack: [6, 4, 3, 2, 1, 5]
# connected pieces: picture graph 1 | graph with three separate parts 3
# bridge counts at each land mass: {'A': 5, 'B': 3, 'C': 3, 'D': 3} | odd ones: 4 | walk using every bridge once exists: False
# after removing one A-B bridge: {'A': 4, 'B': 2, 'C': 3, 'D': 3} | odd ones: 2 | walk exists: True
# maze: BFS shortest 6 steps | brute force over every simple path: 6 | first path found by DFS: 10 steps
```
Reading the output (Python 3.9, standard library only):
- **The graph:** the adjacency list has 14 entries and the matrix 36 cells. The degrees add up to 14, which is twice the 7 edges.
- **BFS:** from vertex 6 it visits 6, 4, 3, 5, 2, 1, the same order as the animation. Distances are 0, 1, 2, 2, 3, 3. The work is 6 vertex visits plus 14 edge looks, which is 20 and equals $V+2E$.
- **Matrix powers:** the distances found by the smallest power with a nonzero entry equal the BFS distances. The $A^2$ entry from 3 to 5 is 2, and a brute-force count of 2-step walks from 3 to 5 also gives 2. The $A^2$ entry from 4 to 4 is 3, which is the degree of vertex 4.
- **DFS:** the recursive and stack versions visit 6, 4, 3, 2, 1, 5 (a different order from BFS). The picture graph has 1 piece and a graph built with three separate parts has 3.
- **Königsberg:** the degrees are 5, 3, 3, 3 (four odd) and no walk uses each bridge once. With one A-B bridge removed, the degrees are 4, 2, 3, 3 (two odd) and such a walk exists.
- **Maze:** BFS gives 6 steps, a check of every route gives 6, and the first route DFS finds is 10 steps.
- **Limits:** tiny inputs, nothing timed.

## 🔴 4. Deep / formal

**Counting edge ends.** Every edge has two ends. Add up the degrees of all vertices and each edge is counted once at each end:

$$\sum_{v\in V}\deg(v)=2\,|E|$$

**Where:**
- $V$ is the set of vertices, and $\deg(v)$ is the number of edges at vertex $v$. **Matters:** the degree is what the list for $v$ holds in an adjacency list.
- $|E|$ is the number of edges. **Matters:** it is the number of pairs of linked vertices (this version assumes no edge joins a vertex to itself, and allows several edges between the same two vertices, as in Königsberg).
- The factor 2 is because each edge touches two vertices. **Matters:** it is why an adjacency list of an undirected graph has $2|E|$ entries, not $|E|$.
- A consequence: the sum is even, so the number of vertices with *odd* degree is always even (0, 2, 4, and so on). **Matters:** this is the key to the Königsberg problem below.

**Example (tiny numbers):** the picture graph, with 7 edges.
1. Degrees of vertices 1 to 6: 2, 3, 2, 3, 3, 1. Their sum is $2+3+2+3+3+1=14$.
2. $2\,|E|=2\times7=14$. The code gave 14 for both.
3. Meaning: the lists in the adjacency list hold 14 items in total. Four vertices have an odd degree (3, 3, 3 and 1), an even number, as the rule says.

**Königsberg and Euler's rule (stated from memory, tested here only on this case).** A walk that uses every edge exactly once, and may start and end in different places, can exist in a connected graph only if the number of odd-degree vertices is 0 or 2. The reason is simple: each time the walk passes *through* a vertex it uses one edge to enter and one to leave, so only the start and the end can have an odd number of edges used. The code's search confirmed the rule for the 7 bridges (4 odd vertices, no walk) and for the 6-bridge version (2 odd vertices, a walk exists). I did not prove the opposite direction (that 0 or 2 odd vertices is *enough* for a walk to exist) and I only checked these two graphs.

**The work of BFS.** BFS takes each vertex off the queue once, and scans its list of neighbours once. The total work is:

$$T_{\text{BFS}}=V+\sum_{v}\deg(v)=V+2E$$

**Where:**
- $V$ is the number of vertex visits (each vertex enters the queue at most once). **Matters:** the "already seen" check is what stops the search from running in circles; without it a graph with a cycle would be searched forever.
- $\sum_v\deg(v)$ is the number of edge looks: each vertex scans its whole list once. **Matters:** by the rule above it equals $2E$ for an undirected graph, so the work grows in step with the size of the graph, not with $V^2$.
- The total, $O(V+E)$, is why BFS and DFS are fast even on huge sparse graphs. **Matters:** with an adjacency matrix instead of lists, each vertex would scan a whole row of $V$ cells, giving $V^2$.

**Example (tiny numbers):** the picture graph, start at 6.
1. Vertex visits: 6. Edge looks: 14 (each of the 7 edges is looked at from both ends).
2. $T=6+14=20$, and $V+2E=6+2\times7=20$. The code counted 20.
3. Meaning: for a graph of this size the search does about 20 small steps. For a graph with a million vertices and 3 million edges it would be about 7 million steps (from the formula, not run), which is why graph search scales to big maps and networks.

**Walks and matrix powers.** Let $A$ be the adjacency matrix: $A_{ij}=1$ if there is an edge between $i$ and $j$ and 0 otherwise. The number of walks of exactly $k$ steps from $i$ to $j$ (a walk may revisit vertices) is the entry in row $i$ and column $j$ of the $k$-th power of $A$:

$$\big(A^{k}\big)_{ij}=\text{the number of walks of length }k\text{ from }i\text{ to }j$$

**Where:**
- $A$ is the matrix of 0s and 1s (see [[Matrix]]). **Matters:** one multiplication by $A$ extends every walk by one step.
- $A^k$ means $A$ multiplied by itself $k$ times. **Matters:** the entry $(A^2)_{ij}=\sum_m A_{im}A_{mj}$ adds up, over every middle vertex $m$, "an edge $i$ to $m$" times "an edge $m$ to $j$". Each such pair is one 2-step walk, so the sum counts them.
- $i,j$ are the start and end vertices. **Matters:** the entry is the count for that pair only.
- $k$ is the number of steps. **Matters:** the smallest $k$ for which the entry is not zero is the shortest distance between $i$ and $j$. This is how the code re-checked the BFS distances.

**Example (tiny numbers):** the picture graph.
1. Walks of 2 steps from 3 to 5: vertex 3 touches 2 and 4, and both 2 and 4 touch 5, so the walks are 3-2-5 and 3-4-5. That makes 2. The code's matrix entry gave 2 and a brute-force count gave 2.
2. Walks of 2 steps from 4 back to 4: go to any neighbour and return. Vertex 4 has 3 neighbours, so there are 3. The matrix entry gave 3, the degree of vertex 4.
3. Meaning: matrix multiplication counts all routes at once. The same idea underlies the spread of influence over a network, and it links graphs to the linear algebra in [[Matrix]] and [[Eigenvalue]] (I only touch on this link here).

**Other points (from memory, not checked here):**
- Leonhard Euler solved the Königsberg problem in 1736, and this is usually called the start of graph theory. I did not check the date, and the map in the picture is an old engraving I did not date.
- Dijkstra's algorithm (1959, from memory) finds shortest routes in weighted graphs and has its own note, [[Dijkstra's Algorithm]]. A* adds a guess of the distance to the goal. I did not cover or run A*.
- Real road and web graphs have millions or billions of vertices, and they are stored in more compact forms than a dictionary of lists.
- Topological order, minimum spanning trees and network flow are other classic graph topics that I did not cover.

**What came before, and how this improved on it:**
- **Before:** special structures for special shapes, such as a list for a line and a tree for a hierarchy.
- **Its limit:** roads, friendships and web links do not fit a line or a tree.
- **What changed:** a graph of nodes and edges, with two ways to store it. Euler's solution of the Königsberg bridges in 1736 is usually named as the start of graph theory (from memory, not checked).
- **How much better:** in the code below, for 10,000 nodes and 30,000 edges an adjacency matrix needs 100,000,000 cells and an adjacency list about 70,000 entries, which is 1,428 times fewer. A breadth-first search of a random graph of 2,000 nodes reached 1,995 of them, the farthest 7 steps away, in 1,995 visits, one visit per node.

**Check it yourself:**

```python
V, E = 10000, 30000
matrix = V * V
lst = V + 2 * E
print(f"graph with {V:,} nodes and {E:,} edges")
print(f"  adjacency matrix: {matrix:,} cells")
print(f"  adjacency list:   about {lst:,} entries ({matrix // lst:,}x fewer)")
from collections import deque
import random
random.seed(0)
adj = {i: set() for i in range(2000)}
for _ in range(6000):
    a, b = random.sample(range(2000), 2); adj[a].add(b); adj[b].add(a)
dist = {0: 0}; q = deque([0])
while q:
    u = q.popleft()
    for v in adj[u]:
        if v not in dist: dist[v] = dist[u] + 1; q.append(v)
print(f"random graph, 2,000 nodes: breadth-first search from node 0 reaches {len(dist):,} nodes, farthest {max(dist.values())} steps away, in {len(dist)} visits")
```

## Related
[[Stacks and Queues]] · [[Arrays and Lists]] · [[Recursion]] · [[Big-O]] · [[Hash Tables]] · [[Dynamic Programming]] · [[Matrix]] · [[Eigenvalue]] · [[Neural Network]] · [[Backpropagation]] · [[Decision Tree]] · [[Vector Database]] · [[Q-Learning]] · [[Python]] · [[Heaps and Priority Queues]] · [[Trees]] · [[Dijkstra's Algorithm]] · [[Beam Search]]
- [[Finite Automaton]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, chapters on elementary graph algorithms (from memory; I did not open it for this note).
- Euler's 1736 paper on the Königsberg bridges and Dijkstra's 1959 paper (named from memory; I did not open either).
- Python documentation for `collections.deque` (docs.python.org).
- Figures: [Breadth-first_search_Algorithm.gif](https://commons.wikimedia.org/wiki/File:Breadth-first_search_Algorithm.gif) (Aexpedito, CC BY-SA 3.0) and [Bridges_of_Konigsberg.png](https://commons.wikimedia.org/wiki/File:Bridges_of_Konigsberg.png) (Twotwos, CC BY-SA 4.0), on Wikimedia Commons. I viewed 7 of the 13 frames of the first and all of the second.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/data-structures Graph:::Dots (vertices) with lines (edges) between linked things. Edges can have a direction (a one-way street) or a weight (the length of a road).

#flashcards/algorithms/concept #flashcards/topic/data-structures Adjacency list against adjacency matrix?::An adjacency list holds, for every vertex, a list of the vertices it touches. An adjacency matrix is a square table with a 1 in row $i$, column $j$ if $i$ and $j$ are linked and a 0 if not.

#flashcards/algorithms/concept #flashcards/topic/data-structures How does the choice of to-do list change graph search?::Taking the oldest item first (a queue) spreads out evenly like ripples: breadth-first search, which finds the route with the fewest edges. Taking the newest first (a stack) goes deep along one path before backing up: depth-first search.

#flashcards/algorithms/concept #flashcards/topic/data-structures How do you count the pieces of a graph?::If a search from one vertex does not reach everything, the graph is in separate pieces (connected components).

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the handshake rule and explain each symbol.
?
$$\sum_{v\in V}\deg(v)=2\,|E|$$
- $V$: the set of vertices. $\deg(v)$: the number of edges at vertex $v$, which is what the list for $v$ holds in an adjacency list.
- $|E|$: the number of edges (this version assumes no edge joins a vertex to itself and allows several edges between the same two vertices, as in Königsberg).
- The factor 2: each edge touches two vertices. So an adjacency list of an undirected graph has $2|E|$ entries, not $|E|$.
- A consequence: the number of odd-degree vertices is always even.

#flashcards/algorithms/example #flashcards/topic/data-structures The picture graph has 7 edges and degrees 2, 3, 2, 3, 3, 1. Check the rule.::Sum $2+3+2+3+3+1=14$ and $2|E|=14$ (the code gave 14 for both). The lists hold 14 items in total. Four vertices have odd degree (3, 3, 3 and 1), an even number, as the rule says.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the running time of breadth-first search and explain each symbol.
?
$$T_{\text{BFS}}=V+\sum_{v}\deg(v)=V+2E$$
- $V$: the number of vertex visits (each vertex enters the queue at most once). The "already seen" check stops the search from running in circles. Without it a graph with a cycle would be searched forever.
- $\sum_v\deg(v)$: the number of edge looks, each vertex scanning its whole list once. It equals $2E$ for an undirected graph, so the work grows with the size of the graph, not with $V^2$.
- The total $O(V+E)$ is why BFS and DFS are fast even on huge sparse graphs.

#flashcards/algorithms/example #flashcards/topic/data-structures Picture graph, start at 6: 6 vertices and 7 edges. What is $T$?::Vertex visits 6, edge looks 14 (each edge looked at from both ends). $T=6+14=20$ and $V+2E=6+14=20$ (the code counted 20). For a million vertices and 3 million edges it would be about 7 million steps (from the formula, not run).

#flashcards/algorithms/formula #flashcards/topic/data-structures Write what the powers of the adjacency matrix count and explain each symbol.
?
$$\big(A^{k}\big)_{ij}=\text{the number of walks of length }k\text{ from }i\text{ to }j$$
- $A$: the matrix of 0s and 1s. One multiplication by $A$ extends every walk by one step.
- $A^k$: $A$ multiplied by itself $k$ times. $(A^2)_{ij}=\sum_m A_{im}A_{mj}$ adds, over every middle vertex $m$, "an edge $i$ to $m$" times "an edge $m$ to $j$". Each such pair is one 2-step walk.
- $i,j$: the start and end vertices. The entry is the count for that pair only.
- $k$: the number of steps. The smallest $k$ for which the entry is not zero is the fewest edges between them.

#flashcards/algorithms/example #flashcards/topic/data-structures In the picture graph, how many 2-step walks go from 3 to 5, and from 4 back to 4?::3 to 5: vertex 3 touches 2 and 4, and both touch 5, so the walks are 3-2-5 and 3-4-5: 2 (matrix entry and brute force agree). 4 to 4: go to any neighbour and return, and vertex 4 has 3 neighbours, so 3, which is the degree of vertex 4.

#flashcards/algorithms/image #flashcards/topic/data-structures ![[bfs-algorithm.gif]]
What does this picture show?
?
Breadth-first search on a graph of 6 vertices, starting at vertex 6. White vertices are not yet seen, grey ones are on the to-do list, black ones are finished. The search spreads out in rings: 6, then 4, then 3 and 5, then 2, then 1. It is an animation, and some viewers show only the first frame. (From the note's caption: Aexpedito, Wikimedia Commons.)

#flashcards/algorithms/image #flashcards/topic/data-structures ![[konigsberg-bridges-map.png]]
What does this picture show?
?
A map of the city of Königsberg with its seven bridges highlighted in pink. The question "can you take a walk that crosses every bridge exactly once?" led to the first result in graph theory. (From the note's caption: Twotwos, Wikimedia Commons. The map itself is an old engraving.)
