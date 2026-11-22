---
tags: [algorithms, graphs, shortest-path, priority-queue, dynamic-programming, programming, python]
status: growing
level: 4
tier: 3
---

# Dijkstra's Algorithm

**Dijkstra's algorithm** finds the **shortest route** from one starting point to every other point in a map where each road has a length (or a cost, or a travel time). It works by always finishing the closest unfinished place next. It is the method behind many route planners, and it is a good example of a [[Graphs]] algorithm that gets its speed from a [[Heaps and Priority Queues]]. It only works when no road has a negative length.

## 🟢 1. Simple (age 12)
You are at home and want the quickest way to every other place in town. First you note how far each *neighbouring* place is. Then you pick the **closest** place you have not finished with, and say "I now know the best way to get here, because any other way would have to go through somewhere even further away". From that place, you check whether going *through it* gives a shorter way to its neighbours. Then you pick the next closest, and so on, until every place is done.

**Example:** You can walk to the shop (4 minutes) or to the park (2 minutes). The park is closest, so you settle on it first. From the park the shop is only 1 more minute. So the shop is now 3 minutes via the park, which beats the direct 4. You update the shop to 3 and carry on.

## 🔵 2. How it works
1. **Start.** Give your starting place the distance 0. Give every other place the distance "infinity" (not reached yet).
2. **Pick the closest unfinished place.** Call it $u$. Its current distance is now final.
3. **Check its roads.** For each road from $u$ to a neighbour $v$ of length $w$, ask: is "the distance to $u$ plus $w$" shorter than what we have for $v$? If yes, update $v$ (equation 1 below). This is called *relaxing* the road.
4. **Repeat** from step 2 until no unfinished place with a known distance is left.
5. **To get the route itself**, remember for each place which place you came from, and walk back from the destination.
6. **Finding the closest unfinished place** is the job of a priority queue (see [[Heaps and Priority Queues]]): put each updated place in with its distance, and take out the smallest.

![[dijkstra-animation.gif]]
*Dijkstra's algorithm on a graph of 6 places (circles), with road lengths on the lines, starting at place 1 and looking for the way to place 5. The small numbers near each circle are the distances found so far. Red circles are finished. The search finishes places in the order 1, 2, 3, 6, 4, 5. This is an animation: some viewers show only the first frame. I viewed 5 of its 43 frames, and the code below uses this same graph. Image: Ibmua, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dijkstra_Animation.gif), public domain.*

![[dijkstra-example-graph.png]]
*A second small weighted graph with five places A to E, used in the code. The number on each line is its length. Image: Shirisha Gongati, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Shortest_path_example_graph.png), CC0. **A warning:** the file's own description says the shortest route from A to E is A, C, D, E with total 12. The weights in the picture, 2, 8 (C to E), 3 and 7, give A, C, E with total 10, which is shorter. I ran the algorithm to check (see the code), and the description is wrong, not the picture. I read the 7 on the line from D to E by zooming in.*

## 🟠 3. Technical detail
- **The rule that makes it work:** once a place is the closest unfinished one, its distance can no longer improve. Any other route to it would have to pass through some unfinished place that is *already at least as far*, and adding a road length that is zero or more cannot make that shorter. This is why **no negative road lengths are allowed**. The code shows a graph with one negative road, where Dijkstra says 4 and the true answer (by Bellman–Ford, which allows negative roads) is 3.
- **Distances and routes:** for the animation's graph, the distances from place 1 are 0, 7, 9, 20, 20 and 11 for places 1 to 6. The shortest route to place 5 is 1, 3, 6, 5 (length 20), and to place 4 it is 1, 3, 4 (also 20).
- **Cost** (see [[Big-O]]): with a heap, each road causes at most one push, so the work is $O((V+E)\log V)$ (equation 3 below). The code counted 8 pushes for the animation graph, which has 18 directed road entries (each two-way road is counted twice). Without a heap, scanning for the closest place each time takes about $V^2$ steps in total, which is fine for dense graphs and slow for big sparse ones (that comparison is from memory; I did not run it).
- **Lazy deletion:** Python's `heapq` cannot lower the value of an item already in the heap. The code simply pushes a new, better entry and skips old entries when they come out (the line `if u in done`). That is why the number of pushes can be larger than the number of places.
- **Equal road lengths:** if every road has length 1, Dijkstra gives the same answer as breadth-first search (see [[Graphs]]), which needs only a plain queue.
- **Negative roads:** use Bellman–Ford instead. It repeatedly relaxes every road $V-1$ times and is simpler but slower. The code uses it as a check.
- **Connection to dynamic programming:** the distance to each place is built from the distances to *smaller* subproblems, in an order where each answer is final when computed. That makes Dijkstra a dynamic-programming method in disguise: equation 2 below is a Bellman equation (see [[Dynamic Programming]], where the same name appears for [[Q-Learning]]).
- **Beyond:** A* adds a guess of the remaining distance to head for the goal faster. I did not cover or run it (named from memory).
- **Limits:** the code runs tiny graphs and random graphs of up to 60 places. It checks correctness and counts operations. It does not time anything and does not handle huge road maps. The random graphs may contain repeated roads and roads from a place to itself, and a place that cannot be reached keeps the distance infinity (not tested separately).

The code covers five things: (1) Dijkstra with a heap, on the animation's graph, (2) the five-place picture's graph, (3) a check against Bellman–Ford and against trying every route on 200 random graphs, (4) the optimality check and the heap-push count, (5) a negative road where it fails.

```python
import heapq, random, itertools
# 1) Dijkstra: always finalise the unfinished vertex with the smallest known distance. A heap finds it fast.
def dijkstra(adj, s):
    dist, parent, done, heap = {v: float('inf') for v in adj}, {}, set(), [(0, s)]
    dist[s] = 0; pushes, order = 1, []
    while heap:
        d, u = heapq.heappop(heap)
        if u in done: continue                                      # an out-of-date entry: skip it
        done.add(u); order.append(u)
        for v, w in adj[u]:
            if v not in done and d + w < dist[v]:                    # relaxation: a shorter way to v, through u
                dist[v] = d + w; parent[v] = u; heapq.heappush(heap, (dist[v], v)); pushes += 1
    return dist, parent, order, pushes
def graph(edges, directed=False):
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, []).append((b, w)); adj.setdefault(b, [])
        if not directed: adj[b].append((a, w))
    return adj
def path(parent, s, t):
    p = [t]
    while p[-1] != s: p.append(parent[p[-1]])
    return p[::-1]
# the six-vertex graph of the animation
E1 = [(1, 2, 7), (1, 3, 9), (1, 6, 14), (2, 3, 10), (2, 4, 15), (3, 4, 11), (3, 6, 2), (4, 5, 6), (5, 6, 9)]
g1 = graph(E1); dist, parent, order, pushes = dijkstra(g1, 1)
print('distances from 1:', dict(sorted(dist.items())), '| finished in order', order, '| heap pushes', pushes, '| directed edges', sum(len(v) for v in g1.values()))
print('shortest route 1 to 5:', path(parent, 1, 5), '= length', dist[5], '| 1 to 4:', path(parent, 1, 4), '= length', dist[4])
# the five-vertex graph of the second picture
E2 = [('A', 'C', 2), ('A', 'B', 4), ('B', 'C', 5), ('B', 'D', 10), ('C', 'D', 3), ('C', 'E', 8), ('D', 'E', 7)]
d2, p2, _, _ = dijkstra(graph(E2), 'A'); print('five-vertex picture: A to E is', path(p2, 'A', 'E'), 'length', d2['E'], '| via C and D it would be', d2['C'] + 3 + 7)
# 2) check against two other methods on random graphs: Bellman-Ford and trying every route
def bellman_ford(adj, s):
    dist = {v: float('inf') for v in adj}; dist[s] = 0
    for _ in range(len(adj) - 1):
        for u in adj:
            for v, w in adj[u]:
                if dist[u] + w < dist[v]: dist[v] = dist[u] + w
    return dist
def brute(adj, s):
    best = {v: float('inf') for v in adj}; best[s] = 0
    def go(u, seen, c):
        best[u] = min(best[u], c)
        for v, w in adj[u]:
            if v not in seen: go(v, seen | {v}, c + w)
    go(s, {s}, 0); return best
def random_graph(n, m, rng):
    return graph([(rng.randrange(n), rng.randrange(n), rng.randint(1, 9)) for _ in range(m)] + [(i, i + 1, rng.randint(1, 9)) for i in range(n - 1)])
rng = random.Random(11); ok_bf = ok_br = 0
for _ in range(200):
    n = rng.randint(2, 7); g = random_graph(n, rng.randint(0, 10), rng)
    d = dijkstra(g, 0)[0]; ok_bf += d == bellman_ford(g, 0); ok_br += d == brute(g, 0)
print('random small graphs where Dijkstra equals Bellman-Ford:', ok_bf, 'of 200 | equals trying every route:', ok_br, 'of 200')
# 3) the optimality check: every vertex's distance is the best (neighbour's distance + edge) over its incoming edges
def certificate(adj, dist, s):
    inc = {v: [] for v in adj}
    for u in adj:
        for v, w in adj[u]: inc[v].append((u, w))
    return all(dist[v] == min(dist[u] + w for u, w in inc[v]) for v in adj if v != s and inc[v])
print('distance = min(neighbour distance + edge) holds for the animation graph:', certificate(g1, dist, 1))
# 4) how many heap pushes? never more than (directed edges + 1)
worst = 0
for _ in range(300):
    n = rng.randint(5, 60); g = random_graph(n, rng.randint(n, 6 * n), rng)
    E = sum(len(v) for v in g.values()); p = dijkstra(g, 0)[3]; assert p <= E + 1; worst = max(worst, p / (E + 1))
print('300 random graphs: pushes never exceeded edges + 1; largest ratio', round(worst, 3))
# 5) it breaks with a negative edge: the "finished" distance of a vertex is trusted and never revisited
neg = graph([('s', 'a', 2), ('s', 'b', 3), ('b', 'a', -2), ('a', 't', 2)], directed=True)
print('negative edge: Dijkstra says t =', dijkstra(neg, 's')[0]['t'], '| Bellman-Ford says', bellman_ford(neg, 's')['t'])
# distances from 1: {1: 0, 2: 7, 3: 9, 4: 20, 5: 20, 6: 11} | finished in order [1, 2, 3, 6, 4, 5] | heap pushes 8 | directed edges 18
# shortest route 1 to 5: [1, 3, 6, 5] = length 20 | 1 to 4: [1, 3, 4] = length 20
# five-vertex picture: A to E is ['A', 'C', 'E'] length 10 | via C and D it would be 12
# random small graphs where Dijkstra equals Bellman-Ford: 200 of 200 | equals trying every route: 200 of 200
# distance = min(neighbour distance + edge) holds for the animation graph: True
# 300 random graphs: pushes never exceeded edges + 1; largest ratio 0.333
# negative edge: Dijkstra says t = 4 | Bellman-Ford says 3
```
Reading the output (Python 3.9, standard library only):
- **Animation graph:** distances from 1 are 0, 7, 9, 20, 20, 11. The places are finished in the order 1, 2, 3, 6, 4, 5, the same as the animation. It used 8 heap pushes for 18 directed road entries. The shortest route to 5 is 1, 3, 6, 5 with length 20, and to 4 it is 1, 3, 4 with length 20.
- **Five-place graph:** the shortest route from A to E is A, C, E with length 10. Going through C and D would be 12 (which is the route the file's description names).
- **Checks:** on 200 random small graphs, Dijkstra equals Bellman–Ford in all 200 and equals "try every route" in all 200.
- **Optimality:** every place's distance equals the minimum, over its incoming roads, of the neighbour's distance plus the road length.
- **Heap pushes:** over 300 random graphs the pushes never exceeded the number of directed road entries plus 1. The largest ratio seen was 0.333.
- **Negative road:** Dijkstra gives 4 for place `t`, Bellman–Ford gives the true 3.
- **Limits:** random graphs have small integer lengths (1 to 9), and nothing was timed.

## 🔴 4. Deep / formal

**Relaxing a road.** Let $d(v)$ be the shortest distance found so far to place $v$, and $w(u,v)$ the length of the road from $u$ to $v$. When $u$ is finished, every road out of it is relaxed:

$$d(v)\ \leftarrow\ \min\big(d(v),\ d(u)+w(u,v)\big)$$

**Where:**
- $d(v)$ is the best known distance to $v$ (starting at $\infty$, or 0 for the start). **Matters:** it is the answer being built. It can only go down, and it becomes final when $v$ is the closest unfinished place.
- $d(u)$ is the final distance to $u$. **Matters:** relaxing from a finished place is what lets the distance spread outward from the start.
- $w(u,v)$ is the road length (zero or more). **Matters:** if it could be negative, a place already finished could later be improved, and the "finished is final" argument would fail (the code's negative-road example).
- $\min$ keeps the shorter of the old and new routes. **Matters:** without it, a longer route found later would overwrite a better one.

**Example (tiny numbers):** the animation's graph, start at place 1.
1. At the start $d(1)=0$ and every other $d=\infty$. Relaxing the roads from 1: $d(2)=\min(\infty,0+7)=7$, $d(3)=\min(\infty,0+9)=9$, $d(6)=\min(\infty,0+14)=14$.
2. The closest unfinished places are 2 (distance 7) and then 3 (distance 9). When place 3 is finished, relax its road to 6 (length 2): $d(6)=\min(14,\,9+2)=11$.
3. The code's final distance for place 6 is 11.
4. Meaning: going through place 3 shortened the way to place 6 from 14 to 11. Each relaxation either improves a distance or changes nothing.

**What "done" means.** When the algorithm finishes, the distances satisfy a simple equation: the distance to each place $v$ (other than the start) is the best way of arriving there from some neighbour:

$$d(v)=\min_{(u,v)\in E}\big(d(u)+w(u,v)\big),\qquad d(s)=0$$

**Where:**
- $(u,v)\in E$ means "every road that ends at $v$". **Matters:** the minimum is taken over all the ways of arriving, so the equation says no arrival is better than the one the algorithm used.
- $d(u)+w(u,v)$ is the cost of going to $u$ optimally and then taking the last road. **Matters:** this is the "build a big answer from smaller ones" step of dynamic programming. It is the same shape as the Bellman equation in [[Dynamic Programming]].
- $d(s)=0$ is the base case. **Matters:** it anchors everything; the start is at distance 0 from itself.
- This equation is a *check*, not a procedure: any set of distances that satisfies it (and is reached by actual routes) is the shortest-path answer when the roads are non-negative. **Matters:** the code checked it on the animation's graph. I did not prove that it is also sufficient. That is a standard result I am quoting from memory.

**Example (tiny numbers):** place 4 in the animation's graph. The roads that arrive at 4 come from places 2 (length 15), 3 (length 11) and 5 (length 6). The final distances are $d(2)=7$, $d(3)=9$, $d(5)=20$.
1. From 2: $7+15=22$. From 3: $9+11=20$. From 5: $20+6=26$.
2. $\min(22,20,26)=20$. The code's $d(4)=20$.
3. Meaning: the best way to place 4 is through place 3, which matches the route 1, 3, 4. The other two ways are longer. The check passed for every place in the graph.

**The cost with a heap.** Each place is finished once. Each road $u\to v$ is looked at once (when $u$ is finished), and at most one new heap entry is pushed per road, plus the start. So the number of pushes is at most the number of roads plus 1, and every push and pop costs $O(\log n)$ in the heap size:

$$\text{pushes}\ \le\ E+1,\qquad T=O\big((V+E)\log V\big)$$

**Where:**
- $E$ is the number of directed road entries (a two-way road counts twice). **Matters:** it bounds the work, because each road is relaxed at most once.
- $V$ is the number of places. **Matters:** the finished places are taken out of the heap one at a time, and each removal costs $O(\log)$ of the heap size.
- $\log V$ is the height of the heap. **Matters:** it is the cost of each push or pop (see [[Heaps and Priority Queues]]). Since the heap holds at most $E+1$ entries, a $\log(E+1)$ cost is the same order as $\log V$ for a graph without repeated roads, because $E\le V^2$.
- The $+1$ is the starting entry. **Matters:** it is the first push.
- The bound is a worst case. **Matters:** the code found real runs well under it (at most 0.333 of the bound on 300 random graphs), because an improved distance is not found for every road.

**Example (tiny numbers):** the animation's graph.
1. $V=6$ places and $E=18$ directed road entries (9 two-way roads).
2. The bound is $E+1=19$ pushes. The code counted 8.
3. Meaning: the heap did far fewer than the worst-case number of pushes on this small graph. For a graph with 1,000,000 roads, the bound is about 1,000,001 pushes at a cost of about 20 small steps each (from the formula, not run), which is why route planners can handle big maps.

**Other points (from memory, not checked here):**
- Edsger Dijkstra published the algorithm in 1959, and later described having devised it in about twenty minutes without pencil and paper. I did not check the date or the story.
- With a more advanced heap (a Fibonacci heap), the cost is $O(E+V\log V)$. I did not cover or run it.
- A* uses the same idea with a guess of the remaining distance, and it is the standard for game and robot path planning. I did not cover it.
- Real route planners add many tricks (such as precomputing shortcuts) that I did not cover.
- Bellman–Ford is $O(VE)$. I used it only as a correctness check, and did not time it.

**What came before, and how this improved on it:**
- **Before:** a breadth-first search, which finds the route with the fewest roads and ignores road lengths.
- **Its limit:** the fewest roads is not the cheapest trip when roads have different costs.
- **What changed:** always expand the closest unfinished place next, using a priority queue. Edsger Dijkstra published it in 1959 (from memory, not checked).
- **How much better:** in the code below (300 places, 1,200 roads costing 1 to 10), the average trip cost from node 0 was 17.36 for the fewest-roads route and 13.01 with Dijkstra, about 25% lower. It needs all road costs to be non-negative.

**Check it yourself:**

```python
import heapq, random
from collections import deque
random.seed(1)
n = 300; adj = {i: [] for i in range(n)}
for i in range(1, n): j = random.randrange(i); w = random.randint(1, 10); adj[i].append((j, w)); adj[j].append((i, w))
for _ in range(600):
    a, b = random.sample(range(n), 2); w = random.randint(1, 10); adj[a].append((b, w)); adj[b].append((a, w))
dist = {0: 0}; pq = [(0, 0)]
while pq:
    d, u = heapq.heappop(pq)
    if d > dist[u]: continue
    for v, w in adj[u]:
        if d + w < dist.get(v, 1e9): dist[v] = d + w; heapq.heappush(pq, (d + w, v))
hop = {0: (0, 0)}; q = deque([0])                         # breadth-first: fewest edges, ignores weights
while q:
    u = q.popleft()
    for v, w in adj[u]:
        if v not in hop: hop[v] = (hop[u][0] + 1, hop[u][1] + w); q.append(v)
print(f"{n} nodes, 1,200 weighted roads (1 to 10); average trip cost from node 0 to every other node:")
print(f"  fewest-roads route (ignores weights): {sum(hop[v][1] for v in range(1, n)) / (n - 1):.2f}")
print(f"  Dijkstra:                             {sum(dist[v] for v in range(1, n)) / (n - 1):.2f}")
```

## Related
[[Graphs]] · [[Heaps and Priority Queues]] · [[Dynamic Programming]] · [[Stacks and Queues]] · [[Big-O]] · [[Recursion]] · [[Q-Learning]] · [[Trees]] · [[Python]] · [[Greedy Algorithms]]

## Sources
- Dijkstra, E. W., "A note on two problems in connexion with graphs", *Numerische Mathematik*, 1959 (named from memory; I did not open it).
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, chapters on single-source shortest paths (from memory; I did not open it for this note).
- Python documentation for the `heapq` module (docs.python.org).
- Figures: [Dijkstra_Animation.gif](https://commons.wikimedia.org/wiki/File:Dijkstra_Animation.gif) (Ibmua, public domain) and [Shortest_path_example_graph.png](https://commons.wikimedia.org/wiki/File:Shortest_path_example_graph.png) (Shirisha Gongati, CC0), on Wikimedia Commons. I viewed 5 of the 43 frames of the first and the whole second picture, and zoomed in on the label of the line from D to E.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/graph-algorithms Dijkstra's algorithm:::Find the quickest way from a start place to every other place. Repeatedly pick the closest unfinished place, which is then final, and check whether going through it shortens its neighbours.

#flashcards/algorithms/concept #flashcards/topic/graph-algorithms List the steps of Dijkstra's algorithm.::Give the start distance 0 and every other place infinity. Pick the closest unfinished place $u$ (its distance is now final). Relax each road from $u$ to a neighbour $v$. Repeat until no unfinished place with a known distance is left. To get the route, remember where each place came from and walk back.

#flashcards/algorithms/concept #flashcards/topic/graph-algorithms Which data structure finds the closest unfinished place?::A priority queue (heap): put each updated place in with its distance and take out the smallest.

#flashcards/algorithms/formula #flashcards/topic/graph-algorithms Write the relaxation rule and explain each symbol.
?
$$d(v)\ \leftarrow\ \min\big(d(v),\ d(u)+w(u,v)\big)$$
- $d(v)$: the best known distance to $v$ (starting at $\infty$, or 0 for the start). It can only go down, and becomes final when $v$ is the closest unfinished place.
- $d(u)$: the final distance to $u$. Relaxing from a finished place lets the distance spread outward from the start.
- $w(u,v)$: the road length (zero or more). If it could be negative, a finished place could later be improved and the "finished is final" argument would fail.

#flashcards/algorithms/example #flashcards/topic/graph-algorithms Start at place 1 with roads 1–2 (7), 1–3 (9), 1–6 (14), and 3–6 (2). What is $d(6)$?::Relaxing from 1: $d(2)=7$, $d(3)=9$, $d(6)=14$. When place 3 is finished, relax 3→6: $d(6)=\min(14,9+2)=11$. Going through place 3 shortened the way to place 6 from 14 to 11. The code's final distance is 11.

#flashcards/algorithms/formula #flashcards/topic/graph-algorithms Write the shortest-distance equation and explain each symbol.
?
$$d(v)=\min_{(u,v)\in E}\big(d(u)+w(u,v)\big),\qquad d(s)=0$$
- $(u,v)\in E$: every road that ends at $v$. The minimum is over all ways of arriving, so no arrival is better than the one the algorithm used.
- $d(u)+w(u,v)$: the cost of going to $u$ optimally and then taking the last road. This is the dynamic-programming step of building a big answer from smaller ones, the same shape as the Bellman equation.
- $d(s)=0$: the base case. The start is at distance 0 from itself.

#flashcards/algorithms/example #flashcards/topic/graph-algorithms Roads arriving at place 4 come from 2 (length 15), 3 (length 11) and 5 (length 6), with $d(2)=7$, $d(3)=9$, $d(5)=20$. What is $d(4)$?::From 2: 22. From 3: 20. From 5: 26. $\min=20$ (the code's $d(4)=20$). The best way to place 4 is through place 3, route 1, 3, 4.

#flashcards/algorithms/formula #flashcards/topic/graph-algorithms Write the cost bounds of Dijkstra with a heap and explain each symbol.
?
$$\text{pushes}\ \le\ E+1,\qquad T=O\big((V+E)\log V\big)$$
- $E$: the number of directed road entries (a two-way road counts twice). Each road is relaxed at most once, which bounds the work.
- $V$: the number of places. Finished places come out of the heap one at a time, each removal costing $O(\log)$ of the heap size.
- $\log V$: the height of the heap, the cost of each push or pop. The heap holds at most $E+1$ entries, so $\log(E+1)$ has the same order as $\log V$ for a graph without repeated roads.

#flashcards/algorithms/example #flashcards/topic/graph-algorithms The animation's graph has $V=6$ places and $E=18$ road entries. What is the push bound and what did the code count?::Bound $E+1=19$ pushes. The code counted 8. For 1,000,000 roads the bound is about 1,000,001 pushes at about 20 small steps each (from the formula, not run), which is why route planners handle big maps.

#flashcards/algorithms/image #flashcards/topic/graph-algorithms ![[dijkstra-animation.gif]]
What does this picture show?
?
Dijkstra's algorithm on a graph of 6 places, with road lengths on the lines, starting at place 1 and looking for the way to place 5. The small numbers near each circle are the distances found so far. Red circles are finished. The search finishes places in the order 1, 2, 3, 6, 4, 5. It is an animation, and some viewers show only the first frame. (From the note's caption: Ibmua, Wikimedia Commons.)

#flashcards/algorithms/image #flashcards/topic/graph-algorithms ![[dijkstra-example-graph.png]]
What does this picture show?
?
A second small weighted graph with five places A to E, used in the note's code. The number on each line is its length. The file's own description says the shortest route from A to E is A, C, D, E with total 12, but the note ran the algorithm and found A, C, E with total 10 from the picture's weights, so the file description is wrong. (From the note's caption: Shirisha Gongati, Wikimedia Commons, CC0.)

#flashcards/algorithms/cloze #flashcards/topic/graph-algorithms Dijkstra's algorithm needs road lengths ==zero or more==. Negative lengths break the "finished is final" argument.
