---
tags: [algorithms, data-structures, stack, queue, programming, python]
status: growing
level: 4
tier: 1
---

# Stacks and Queues

A **stack** and a **queue** are two simple ways of holding a collection of things, and the only difference is *which item comes out first*. In a stack, the **last** item put in comes out first (like a pile of plates). In a queue, the **first** item put in comes out first (like a line at a shop). Both are usually built on top of an array or a linked list (see [[Arrays and Lists]]), and both appear everywhere: the "undo" button, the way programs remember where to return to after a function call (see [[Recursion]]), and the way a computer decides what to do next.

## 🟢 1. Simple (age 12)
A **stack** is like a pile of plates. You put a clean plate on top, and you take a plate from the top. The plate you put on last is the first one you pick up. You cannot easily pull one out of the middle.

A **queue** is like the line at a school canteen. The person who joined first is served first, and new people join at the back. Nobody is allowed to push in.

**Example:** The back button in a web browser is a stack. Each page you visit is put on the pile, and "back" takes the top page off, which is the one you visited most recently. A printer works like a queue: documents are printed in the order they were sent.

## 🔵 2. How it works
1. **A stack has two main actions.** *Push* puts an item on top. *Pop* takes the top item off. You only ever touch the top.
2. **A queue has two main actions.** *Enqueue* adds an item at the back. *Dequeue* removes the item at the front. You add at one end and remove at the other.
3. **Both can be built from a list.** For a stack, use the end of a list for both push and pop. This is fast. A queue is trickier: removing from the front of a plain list is slow, because all the other items shift along (see [[Arrays and Lists]]). So Python provides `collections.deque`, which is fast at both ends.
4. **A queue can reuse its space in a circle.** A fixed block of memory is treated as a ring. A "head" mark shows where the next item leaves and a "tail" mark shows where the next item arrives. When a mark reaches the end of the block, it wraps around to the start. This is called a *ring buffer* or *circular buffer*.
5. **Why they matter:** the same walk through a collection of connected things (see [[Graphs]]) can be done in two different ways, depending only on whether you keep the "to do next" items in a stack or in a queue. The code shows this.

![[queue-vs-stack.svg]]
*A queue (left) and a stack (right). In the queue, items are added at one end ("Enqueue") and removed from the other ("Dequeue"), so the first in is the first out (FIFO). In the stack, items are pushed and popped at the same end, so the last in is the first out (LIFO). Image: Choteseeworld, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:QUEUE_VS_STACK.svg), CC BY-SA 4.0.*

![[circular-buffer-animation.gif]]
*A circular buffer used as a keyboard queue. Letters typed are written at the red "write" pointer, and the processor reads them at the blue "read" pointer. Both pointers move round the ring and wrap past the end back to the start. This is an animation: some viewers show only the first frame. I viewed two of its 32 frames (the 11th and the last): letters of a typed phrase fill the ring, and in the last frame both pointers sit at the same place. Image: MuhannadAjjan, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Circular_Buffer_Animation.gif), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Cost of the operations** (see [[Big-O]]): push, pop, enqueue and dequeue are all $O(1)$ when built well. A Python `list` is a good stack. For a queue use `collections.deque` (`append` and `popleft`). Using `list.pop(0)` for a queue is $O(n)$ per removal, because the rest shifts (see [[Arrays and Lists]]).
- **Stack uses:**
  - *Matching brackets:* push each opening bracket, and each closing bracket must match the top. The code checks four strings.
  - *Evaluating expressions:* in *postfix* (reverse Polish) notation, `3 4 + 2 *` means $(3+4)\times2$. Push numbers, and when you meet an operator, pop two numbers and push the result. The code evaluates this to 14.
  - *The call stack:* each running function call is a frame on a stack. This is why very deep recursion fails (see [[Recursion]]) and why a recursive walk can be rewritten with a stack of your own.
  - *Undo, backtracking, parsing:* anything that must return to "the most recent unfinished thing".
- **Queue uses:** a line of jobs, messages passed between programs, and *breadth-first search*. The code walks the same small tree twice. With a queue, it visits level by level: a, b, c, d, e, f. With a stack, it goes deep first: a, b, d, e, c, f. Only the end the next item is taken from changes. The graph version is in [[Graphs]].
- **Ring buffer:** a queue in a fixed block of memory with no shifting. The next free place is $(\text{head}+\text{size})\bmod\text{capacity}$ (equation 1 below). The code checks it against `deque` over 10,000 random operations. It must also handle a *full* queue (the code raises an error).
- **Waiting lines as maths:** how long customers wait in a single line follows a simple recurrence (equation 2), and the average number of customers in the system equals the arrival rate times the average time each one spends there (equation 3, *Little's law*).
- **Other related structures:** a *priority queue* (the item with the highest priority leaves first, not the oldest) and a *deque* (both ends allowed). I do not cover priority queues in depth here.
- **Limits:** the code counts and checks results. It does not time anything and does not test thread safety (using one queue from several threads at once, which is a separate topic).

The code covers five things: (1) bracket matching and postfix evaluation with a stack, (2) a ring-buffer queue checked against `deque`, (3) the same walk with a queue and with a stack, (4) one-line waiting times by a recurrence and by simulation, (5) Little's law, with a tiny example and 20,000 customers.

```python
import random
from collections import deque
# 1) a stack: last in, first out. A list is a fine stack (append = push, pop = pop)
def balanced(text):
    pairs, stack = {')': '(', ']': '[', '}': '{'}, []
    for ch in text:
        if ch in '([{': stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]: return False
    return not stack
for t in ('(a[b]{c})', '(a[b)]', '((', ')('): print(repr(t), '->', balanced(t))
def postfix(tokens):                                         # "3 4 + 2 *" means (3 + 4) * 2
    st = []
    for tok in tokens.split():
        if tok in '+-*/': b, a = st.pop(), st.pop(); st.append({'+': a + b, '-': a - b, '*': a * b, '/': a / b}[tok])
        else: st.append(float(tok))
    return st[0]
print('3 4 + 2 * =', postfix('3 4 + 2 *'), '| 5 1 2 + 4 * + 3 - =', postfix('5 1 2 + 4 * + 3 -'))
# 2) a queue: first in, first out. deque is the right tool; here is a fixed-size ring buffer to show the index trick
class Ring:
    def __init__(self, cap): self.buf, self.head, self.size, self.cap = [None] * cap, 0, 0, cap
    def push(self, x):
        if self.size == self.cap: raise OverflowError('full')
        self.buf[(self.head + self.size) % self.cap] = x; self.size += 1          # tail = (head + size) mod capacity
    def pop(self):
        x = self.buf[self.head]; self.head = (self.head + 1) % self.cap; self.size -= 1; return x
r = Ring(4)
for x in 'abc': r.push(x)
r.pop(); r.pop(); r.push('d'); r.push('e'); r.push('f')
print('ring after wrap-around: buf', r.buf, '| head', r.head, '| size', r.size, '| pops give', ''.join(r.pop() for _ in range(r.size)))
random.seed(1); q, d, ok = Ring(8), deque(), True
for _ in range(10000):
    if d and (len(d) == 8 or random.random() < 0.5): ok &= q.pop() == d.popleft()
    else: v = random.random(); q.push(v); d.append(v)
print('ring buffer agrees with deque over 10000 random operations:', ok)
# 3) the same search with a stack or a queue visits in a different order
tree = {'a': ['b', 'c'], 'b': ['d', 'e'], 'c': ['f'], 'd': [], 'e': [], 'f': []}
def walk(start, queue):
    todo, order = deque([start]), []
    while todo:
        node = todo.popleft() if queue else todo.pop()
        order.append(node); todo.extend(reversed(tree[node]) if not queue else tree[node])
    return order
print('queue (breadth first):', walk('a', True), '| stack (depth first):', walk('a', False))
# 4) one server, one line: Lindley's recurrence W_next = max(0, W + service - gap), checked against an event-by-event simulation
def lindley(gaps, services):
    w, out = 0, [0]
    for g, s in zip(gaps[1:], services): w = max(0, w + s - g); out.append(w)
    return out
gaps, services = [0, 2, 1, 1, 3], [3, 2, 2, 1, 1]                   # gap[i] = time since previous arrival
print('tiny example waits:', lindley(gaps, services))
def simulate(gaps, services):
    t, free, waits, done = 0, 0, [], []
    for g, s in zip(gaps, services):
        t += g; start = max(t, free); waits.append(start - t); free = start + s; done.append((t, free))
    return waits, done
w1, done_tiny = simulate(gaps, services); print('event simulation waits:', w1, '| same:', w1 == lindley(gaps, services))
random.seed(7); n = 20000; G = [0] + [random.randint(1, 5) for _ in range(n - 1)]; S = [random.randint(1, 4) for _ in range(n)]
w2, done = simulate(G, S); print('20000 customers: Lindley equals simulation:', lindley(G, S) == w2, '| mean wait', round(sum(w2) / n, 3))
# 5) Little's law: average number in the system L = arrival rate x average time in the system W
def little(done):
    T = done[-1][1]; n = len(done); stay = [e - a for a, e in done]
    area, cur, last = 0, 0, 0
    for t, d_ in sorted([(a, 1) for a, _ in done] + [(e, -1) for _, e in done]): area += cur * (t - last); cur += d_; last = t    # time-average count by walking the events
    return round(area / T, 4), round(n / T * sum(stay) / n, 4), sum(stay), T
print('tiny example (area / T, lambda * W, total stay, T):', little(done_tiny))
print('20000 customers (area / T, lambda * W, total stay, T):', little(done))
# '(a[b]{c})' -> True
# '(a[b)]' -> False
# '((' -> False
# ')(' -> False
# 3 4 + 2 * = 14.0 | 5 1 2 + 4 * + 3 - = 14.0
# ring after wrap-around: buf ['e', 'f', 'c', 'd'] | head 2 | size 4 | pops give cdef
# ring buffer agrees with deque over 10000 random operations: True
# queue (breadth first): ['a', 'b', 'c', 'd', 'e', 'f'] | stack (depth first): ['a', 'b', 'd', 'e', 'c', 'f']
# tiny example waits: [0, 1, 2, 3, 1]
# event simulation waits: [0, 1, 2, 3, 1] | same: True
# 20000 customers: Lindley equals simulation: True | mean wait 2.384
# tiny example (area / T, lambda * W, total stay, T): (1.7778, 1.7778, 16, 9)
# 20000 customers (area / T, lambda * W, total stay, T): (1.6336, 1.6336, 97689, 59800)
```
Reading the output (Python 3.9, standard library only):
- **Brackets:** `(a[b]{c})` is balanced. `(a[b)]`, `((` and `)(` are not.
- **Postfix:** both expressions give 14.
- **Ring buffer:** after pushing `a b c`, popping two, and pushing `d e f` into 4 slots, the contents are `['e','f','c','d']` with the head at 2 and 4 items, and the pops come out in the order `c d e f`: the new items wrapped round to the front of the block. It agrees with `deque` over 10,000 random operations.
- **Walks:** the queue gives a, b, c, d, e, f, and the stack gives a, b, d, e, c, f.
- **Waiting line:** the tiny example's waits are 0, 1, 2, 3, 1 by the recurrence and by the event simulation. For 20,000 random customers (gaps 1 to 5, services 1 to 4), the recurrence equals the simulation exactly, and the mean wait is 2.384 time units.
- **Little's law:** for the tiny example, the average number in the system measured from the events is 1.7778, and the arrival rate times the average stay is also 1.7778. For 20,000 customers both are 1.6336.
- **Limits:** only counts and exact checks, nothing timed.

## 🔴 4. Deep / formal

**Where the next item goes in a ring buffer.** A queue lives in a block of $c$ slots (the capacity). The head is the slot of the oldest item and $m$ is how many items are inside. The next item is written into the slot:

$$\text{tail}=(\text{head}+m)\bmod c$$

**Where:**
- $\text{head}$ is the index of the oldest item (the next one to leave). **Matters:** it is the only mark you need to read from; popping just moves it on by one (also modulo $c$).
- $m$ is the current number of items. **Matters:** the new item goes $m$ places after the head, because the $m$ items occupy the places head, head+1, and so on.
- $c$ is the capacity of the block. **Matters:** it fixes the size of the ring. The queue is full when $m=c$, and writing more would overwrite the oldest item, so the code raises an error.
- $\bmod c$ means "the remainder after dividing by $c$". **Matters:** this is the wrap-around. Without it the index would run off the end of the block, and you would have to shift the items back to the start (the slow way).

**Example (tiny numbers):** $c=4$. In the code, after two pops the head is at 2, and after pushing $d$ and $e$ the queue holds $c$, $d$, $e$ ($m=3$). Now $f$ arrives.
1. $\text{tail}=(2+3)\bmod4=5\bmod4=1$.
2. So the next item ($f$) goes to slot 1. The code's final block is `['e','f','c','d']`, with $f$ in slot 1.
3. Meaning: slots 2, 3, 0 hold $c,d,e$, and the next free slot is 1, which is *before* the head in memory but after it in the ring. No item ever moved.

**How long you wait in a single line.** One server, customers served in order of arrival. Let $W_k$ be how long customer $k$ waits before service starts, $S_k$ the service time of customer $k$, and $A_{k+1}$ the time between the arrival of customer $k$ and customer $k+1$ (the "gap"). Then:

$$W_{k+1}=\max\big(0,\ W_k+S_k-A_{k+1}\big),\qquad W_1=0$$

**Where:**
- $W_k$ is the wait of customer $k$ (time in the line before being served). **Matters:** it is the quantity we track, and the next customer's wait is built from it.
- $S_k$ is how long serving customer $k$ takes. **Matters:** a long service makes everyone behind wait longer.
- $A_{k+1}$ is the gap before the next arrival. **Matters:** a long gap gives the server time to catch up. A short gap piles up the line.
- $W_k+S_k-A_{k+1}$ is how long the next customer would have to wait: the time until customer $k$ finishes (wait plus service, counted from customer $k$'s arrival), minus the time that has already passed before the next one arrives.
- $\max(0,\cdot)$ means a wait can never be negative. **Matters:** if the server finishes before the next customer arrives, the server is idle and the next customer is served at once (wait 0). Without the $\max$, idle time would wrongly be treated as "negative waiting".

**Example (tiny numbers):** the code's example, gaps 0, 2, 1, 1, 3 and services 3, 2, 2, 1, 1.
1. $W_1=0$.
2. $W_2=\max(0,\,0+3-2)=1$.
3. $W_3=\max(0,\,1+2-1)=2$. $W_4=\max(0,\,2+2-1)=3$. $W_5=\max(0,\,3+1-3)=1$.
4. The recurrence gives waits 0, 1, 2, 3, 1, which is exactly what the event simulation gave.
5. Meaning: each customer's wait depends only on the one before. Customers 2 to 4 arrive faster than the server can finish, so the wait grows (1, 2, 3). Then a longer gap (3) lets it shrink again.

**Little's law.** Watch a system (a shop, a queue, a server) from the time it is empty until it is empty again. Let $L$ be the average number of customers in it, $\lambda$ the average arrival rate and $W$ the average time one customer spends in it:

$$L=\lambda\,W$$

**Where:**
- $L$ is the average number of customers in the system over the whole period, from the area under the "number present" curve divided by the length of the period. **Matters:** it is the "how crowded is it" measure.
- $\lambda$ is arrivals per unit time ($n/T$ for $n$ customers over a period $T$). **Matters:** more arrivals mean a more crowded system for the same stay.
- $W$ is the average total time one customer spends in the system (waiting plus being served). **Matters:** longer stays mean a more crowded system for the same arrival rate.
- In this finite run it is an exact identity: both sides equal (total of all stays) divided by $T$. **Matters:** the area under the count curve is the same as the sum of everyone's stays, counted two ways. For long-run averages of random systems the law also holds under mild conditions, which I did not look into (from memory).

**Example (tiny numbers):** the five-customer example. Arrivals at 0, 2, 3, 4, 7 and departures at 3, 5, 7, 8, 9.
1. Stays: 3, 3, 4, 4, 2. Total 16. The period is $T=9$, so $\lambda=5/9$ and $W=16/5=3.2$.
2. $\lambda W=\frac59\times3.2=\frac{16}{9}=1.7778$.
3. Counting from the events instead (the area under the curve of "number present") gives $16/9=1.7778$ too. The code gave 1.7778 both ways.
4. Meaning: on average about 1.8 customers were in the shop at any moment. Knowing the arrival rate and the average stay is enough to know the crowding, without counting heads at every moment. For 20,000 customers both sides were 1.6336.

**Other points (from memory, not checked here):**
- John Little proved the law in 1961, and it is named after him. I did not check the date or the paper.
- The recurrence for waiting times is usually credited to Lindley (1952). I did not check this either, and my name for the recurrence ("Lindley's") comes from memory.
- Queueing theory studies waiting lines with random arrivals and services in much more depth. I did not cover the standard formulas for the average wait (they need assumptions about the random model).
- Implementations of ring buffers differ in how they tell "full" from "empty" (a spare slot, or a separate count). I used a count.

**What came before, and how this improved on it:**
- **Before:** use a plain list for everything and take items from wherever you like.
- **Its limit:** taking from the front of a list shifts every other item left, which is slow for long lists.
- **What changed:** restrict yourself to two simple disciplines, last in first out (stack) and first in first out (queue), and use a structure built for them such as a double-ended queue.
- **How much better:** in the code below, taking 10,000 items off the front of a list one by one costs 49,995,000 shifts, and with a deque 10,000 steps, which is 4,999 times fewer. The same snippet uses a stack to check brackets: `{[()()]}` is balanced and `{[(])}` is not.

**Check it yourself:**

```python
from collections import deque
n = 10000
shifts = sum(n - 1 - i for i in range(n))               # list.pop(0) shifts everything left
print(f"taking {n:,} items off the front of a list one by one: {shifts:,} shifts; with a deque: {n:,} steps ({shifts // n:,}x fewer)")
def balanced(s):
    st, pair = [], {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{": st.append(ch)
        elif ch in pair:
            if not st or st.pop() != pair[ch]: return False
    return not st
for s in ("{[()()]}", "{[(])}", "((("):
    print(f"{s!r:12} balanced: {balanced(s)}")
q = deque([1, 2, 3]); q.append(4); print("queue: first in, first out ->", q.popleft(), q.popleft())
```

## Related
[[Arrays and Lists]] · [[Recursion]] · [[Big-O]] · [[Graphs]] · [[Dynamic Programming]] · [[Sorting]] · [[Hash Tables]] · [[Python]] · [[Heaps and Priority Queues]] · [[Trees]]
- [[Finite Automaton]]

## Sources
- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, on stacks and queues (from memory; I did not open it for this note).
- Python documentation for `collections.deque` and `list` (docs.python.org).
- Figures: [QUEUE_VS_STACK.svg](https://commons.wikimedia.org/wiki/File:QUEUE_VS_STACK.svg) (Choteseeworld, CC BY-SA 4.0) and [Circular_Buffer_Animation.gif](https://commons.wikimedia.org/wiki/File:Circular_Buffer_Animation.gif) (MuhannadAjjan, CC BY-SA 4.0), on Wikimedia Commons. I viewed the first, and two of the 32 frames of the second, before writing the captions.
- The code and every Example number were run for this note (Python 3.9, standard library only). Operations were counted, not timed.

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/data-structures Stack:::A pile of plates. The last item put on is the first taken off (LIFO). Push puts an item on top and pop takes the top one. You only touch the top.

#flashcards/algorithms/reverse #flashcards/topic/data-structures Queue:::A canteen line. The first in is the first served (FIFO). Enqueue adds at the back and dequeue removes from the front.

#flashcards/algorithms/concept #flashcards/topic/data-structures How do you build a stack and a queue in Python?::A stack: use the end of a list for both push and pop (fast). A queue: removing from the front of a plain list is slow (everything shifts), so Python provides `collections.deque`, which is fast at both ends.

#flashcards/algorithms/reverse #flashcards/topic/data-structures Ring buffer (circular buffer):::A fixed block of memory treated as a ring. A head mark shows where the next item leaves and a tail mark where the next item arrives. When a mark reaches the end of the block it wraps around to the start.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the slot where the next item goes in a ring buffer and explain each symbol.
?
$$\text{tail}=(\text{head}+m)\bmod c$$
- $\text{head}$: the index of the oldest item (the next one to leave). Popping just moves it on by one (also modulo $c$).
- $m$: the current number of items. The new item goes $m$ places after the head, because the $m$ items occupy head, head+1, and so on.
- $c$: the capacity of the block. It fixes the size of the ring. The queue is full when $m=c$, and writing more would overwrite the oldest item, so the note's code raises an error.
- $\bmod c$: wraps the position around the ring.

#flashcards/algorithms/example #flashcards/topic/data-structures $c=4$, head at 2, the queue holds $c$, $d$, $e$ ($m=3$), and $f$ arrives. Which slot does it use?::$\text{tail}=(2+3)\bmod4=1$. $f$ goes to slot 1 (the code's final block is `['e','f','c','d']`). Slots 2, 3, 0 hold $c,d,e$ and the next free slot is 1, before the head in memory but after it in the ring. No item ever moved.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write the wait of the next customer in a single-server queue and explain each symbol.
?
$$W_{k+1}=\max\big(0,\ W_k+S_k-A_{k+1}\big),\qquad W_1=0$$
- $W_k$: the wait of customer $k$ (time in the line before being served). The next customer's wait is built from it.
- $S_k$: how long serving customer $k$ takes. A long service makes everyone behind wait longer.
- $A_{k+1}$: the gap before the next arrival. A long gap gives the server time to catch up, a short gap piles up the line.
- $W_k+S_k-A_{k+1}$: how long the next customer would wait (the time until customer $k$ finishes, minus the arrival gap). The $\max$ with 0 stops the wait going negative.

#flashcards/algorithms/example #flashcards/topic/data-structures Gaps 0, 2, 1, 1, 3 and services 3, 2, 2, 1, 1. What are the waits?::$W_1=0$. $W_2=\max(0,0+3-2)=1$. $W_3=\max(0,1+2-1)=2$. $W_4=\max(0,2+2-1)=3$. $W_5=\max(0,3+1-3)=1$. Waits 0, 1, 2, 3, 1, exactly what the event simulation gave. Customers 2 to 4 arrive faster than the server finishes, so the wait grows. Then a longer gap lets it shrink.

#flashcards/algorithms/formula #flashcards/topic/data-structures Write Little's law and explain each symbol.
?
$$L=\lambda\,W$$
- $L$: the average number of customers in the system over the whole period (the area under the "number present" curve divided by the period's length). The "how crowded" measure.
- $\lambda$: arrivals per unit time ($n/T$ for $n$ customers over a period $T$). More arrivals mean a more crowded system for the same stay.
- $W$: the average total time one customer spends in the system (waiting plus being served). Longer stays mean a more crowded system for the same arrival rate.

#flashcards/algorithms/example #flashcards/topic/data-structures Arrivals at 0, 2, 3, 4, 7 and departures at 3, 5, 7, 8, 9. What is $L$?::Stays 3, 3, 4, 4, 2, total 16. $T=9$, so $\lambda=5/9$ and $W=16/5=3.2$. $\lambda W=\frac59\times3.2=\frac{16}{9}=1.7778$. Counting from the events (the area under the curve) also gives 1.7778. On average about 1.8 customers were in the shop at any moment.

#flashcards/algorithms/image #flashcards/topic/data-structures ![[queue-vs-stack.svg]]
What does this picture show?
?
A queue (left) and a stack (right). In the queue, items are added at one end (enqueue) and removed from the other (dequeue), so the first in is the first out (FIFO). In the stack, items are pushed and popped at the same end, so the last in is the first out (LIFO). (From the note's caption: Choteseeworld, Wikimedia Commons.)

#flashcards/algorithms/image #flashcards/topic/data-structures ![[circular-buffer-animation.gif]]
What does this picture show?
?
A circular buffer used as a keyboard queue. Letters typed are written at the red write pointer and the processor reads them at the blue read pointer. Both pointers move round the ring and wrap past the end back to the start. It is an animation, and some viewers show only the first frame. (From the note's caption: MuhannadAjjan, Wikimedia Commons.)

#flashcards/algorithms/cloze #flashcards/topic/data-structures Using a stack in graph search gives ==depth-first==, and using a queue gives ==breadth-first==.
