---
tags: [algorithms, theory, automata, regex, state-machine]
status: growing
level: 4
tier: 2
---

# Finite Automaton

A **finite automaton** (also called a *finite state machine*) is a tiny machine with a fixed, small number of **states**. It reads its input one symbol at a time and, after each symbol, moves to a new state by a fixed rule. It remembers nothing except which state it is in.

## 🟢 1. Simple (age 12)
Think of a turnstile at a train station. It is either **locked** or **unlocked**. Drop in a coin and it unlocks. Push through and it locks again. It never needs to remember how many people passed, only which of the two situations it is in right now.

**Example:** a traffic light is a machine with three states (green, yellow, red). The rule for what comes next depends only on the colour it is now.

## 🔵 2. How it works
1. List the **states** the machine can be in. There are only a few, and the list never grows.
2. Mark one state as the **start**.
3. Write down a rule for every pair (current state, next symbol): which state to move to. A table or a drawing with arrows ([[Graphs]]) shows all the rules.
4. Feed in the input one symbol at a time. After each symbol, follow the rule and move.
5. Mark some states as **accepting** ("yes"). When the input ends, the machine says yes if it is in an accepting state and no otherwise.
6. Because the machine only stores the current state, reading a million symbols uses the same tiny memory as reading ten. Each symbol costs one step.
7. **The catch:** a fixed number of states means a fixed amount of memory. It cannot count without limit, so it cannot check that brackets match to any depth.

![[turnstile-state-machine.svg]]
*A finite state machine for a turnstile: two states, Locked and Unlocked (the filled dot marks the start), and two inputs, Coin and Push. A coin unlocks it, a push through the unlocked turnstile locks it again, and the other two arrows leave the state unchanged. Image: Chetvorno, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Turnstile_state_machine_colored.svg), CC0. I read the states and arrows off the picture; the description of the turnstile is the standard textbook one.*

## 🟠 3. Technical detail
- **Parts:** a set of states, an alphabet of input symbols, a transition rule, a start state and a set of accepting states.
- **Deterministic and not:** in a deterministic automaton each (state, symbol) pair has exactly one next state. A non-deterministic one may allow several, and can always be converted into a deterministic one, sometimes with many more states (from memory).
- **Link to regular expressions:** a [[Regular Expressions|regular expression]] and a finite automaton describe exactly the same family of patterns (the "regular languages"; Kleene, from memory). Fast regex engines turn the pattern into a machine, so they read the text once with no backing up.
- **Contrast with backtracking:** an engine that tries one way and then another ([[Backtracking]]) can take exponential time on nasty patterns, while the machine takes one step per symbol.
- **What it cannot do:** anything that needs unbounded memory, such as matching brackets to any depth. A machine with a stack ([[Stacks and Queues]]) handles that, which is one step up in power.
- **Where it is used:** the lexer that splits program text into pieces ([[Tokenization]] is a cousin), text search, network protocols, game character behaviour and vending-machine logic.

```python
# A turnstile: two states, two kinds of input. The table IS the machine.
table = {("locked", "coin"): "unlocked", ("locked", "push"): "locked",
         ("unlocked", "coin"): "unlocked", ("unlocked", "push"): "locked"}
state = "locked"                                   # the start state
for event in ("push", "coin", "coin", "push", "push"):
    state = table[(state, event)]
    print(f"{event:5s} -> {state}")
print("states:", len({s for s, _ in table}), " table entries:", len(table), " memory while running: 1 state name")
```

## 🔴 4. Deep / formal
**One step of a deterministic machine.** The state after reading a symbol is a function of the old state and the symbol:

$$
q_{t+1}=\delta(q_t,\,x_t)
$$

**Where:**
- $q_t$ is the state before reading the $t$-th symbol and $q_{t+1}$ the state after. The machine starts at a chosen start state $q_0$.
- $x_t$ is the $t$-th input symbol.
- $\delta$ (delta) is the transition table. **Matters:** it is all the machine is. There is no other memory, so what the machine "remembers" is exactly what the states can distinguish.
- The input is accepted if the last state is one of the accepting states.

**Example (tiny numbers):** a machine that tells whether a binary number is divisible by 3. Its states are the remainders 0, 1 and 2, and the rule is $\delta(q,b)=(2q+b)\bmod 3$ (reading a bit doubles the number so far and adds the bit). Start at $q_0=0$ and read `110` (which is 6).
- Read 1: $q_1=(2\cdot0+1)\bmod3=\mathbf1$.
- Read 1: $q_2=(2\cdot1+1)\bmod3=3\bmod3=\mathbf0$.
- Read 0: $q_3=(2\cdot0+0)\bmod3=\mathbf0$.
- The last state is 0, an accepting state, so the answer is yes, and 6 is indeed divisible by 3.
- Meaning: three states are enough to remember the one fact that matters, the remainder, however long the number.

**Why states limit counting.** A machine with $k$ states cannot tell apart more than $k$ different situations. To check that brackets balance to depth $d$ it needs to tell apart the depths $0,1,\dots,d$, which takes at least $d+1$ states, and a bound that changes with the input is impossible for a fixed machine. The code at the end shows a 4-state machine that handles depth 3 and fails on a balanced string of depth 4.

**Further notes:**
- **Minimal machines:** every pattern has a smallest machine, and it can be found by merging states that behave the same (from memory).
- **Pushdown automata:** add a stack and you can match brackets. Add a tape that can be read and written freely, and you get a Turing machine (from memory).
- **Not only for text:** the same table idea is used for game characters, traffic lights and protocol handlers.

**What came before, and how this improved on it:**
- **Before:** ad hoc code that stores the whole input and works on it, or tangled if-then rules for every case.
- **Its limit:** the memory grows with the input, and the rules are hard to check and change.
- **What changed:** state the memory you need as a fixed set of states and write the behaviour as one table. McCulloch and Pitts modelled neurons this way in 1943 (checked on the Wikipedia timeline page, see [[History of AI]]), Kleene described regular events in 1951 and Rabin and Scott studied the machines in 1959 (both from memory, not checked).
- **How much better:** in the code below, a 3-state machine for "divisible by 3" agreed with the arithmetic check `int(s, 2) % 3 == 0` on all 100,000 random binary strings, while keeping 1 number in memory instead of up to 40 characters. The same pattern as a regular expression matched exactly the multiples of 3 for 0 to 1023. Cutting a text of 20 letters into blocks has 524,288 ways (the idea behind a backtracking slowdown, see [[Backtracking]]), while the machine takes 20 steps. **Honest limit:** the 4-state bracket machine accepted `((()))` but rejected the balanced `(((())))`, because it ran out of states.

**Check it yourself:**

```python
import random, re
delta = {(q, b): (2 * q + b) % 3 for q in range(3) for b in (0, 1)}   # state = remainder of the number read so far, divided by 3
def dfa(s):
    q = 0
    for ch in s: q = delta[(q, int(ch))]
    return q == 0
random.seed(0)
tests = ["".join(random.choice("01") for _ in range(random.randint(1, 40))) for _ in range(100000)]
print("100,000 random binary strings: the 3-state machine agrees with int(s, 2) % 3 == 0 on", sum(dfa(s) == (int(s, 2) % 3 == 0) for s in tests))
print("memory used while reading: the machine keeps 1 number (0, 1 or 2); storing the string keeps up to", max(map(len, tests)), "characters")
rx = re.compile(r"(0|1(01*0)*1)*")
print("the same language as a regular expression, checked on 0..1023:", all(bool(rx.fullmatch(format(i, "b"))) == (i % 3 == 0) for i in range(1024)))
def depth_machine(k):                                                   # a machine with k+1 states can count nesting depth only up to k
    def run(s):
        q = 0
        for ch in s:
            q += 1 if ch == "(" else -1
            if q < 0 or q > k: return False
        return q == 0
    return run
m = depth_machine(3)
print("balanced brackets with a 4-state machine (depth up to 3): '((()))' ->", m("((()))"), " '(((())))' ->", m("(((())))"), "(the second is balanced, but too deep for 4 states)")
```

## Related
- [[Regular Expressions]] · [[Backtracking]] · [[Graphs]] · [[Tokenization]] · [[Stacks and Queues]] · [[Big-O]] · [[Recursion]] · [[History of AI]]

## Sources
- The turnstile image: [Turnstile_state_machine_colored.svg](https://commons.wikimedia.org/wiki/File:Turnstile_state_machine_colored.svg) by Chetvorno, CC0.
- McCulloch and Pitts (1943): date checked on the Wikipedia *Timeline of artificial intelligence* page. Kleene (1951), Rabin and Scott (1959) and the facts about non-deterministic machines, pushdown automata and minimal machines are from memory, not opened.
- All code results were run for this note (plain Python, the standard `re` module).

## Flashcards
#flashcards/algorithms

#flashcards/algorithms/reverse #flashcards/topic/complexity Finite automaton (finite state machine):::A machine with a few states and rules for moving between them. A turnstile is either locked or unlocked and never needs to remember how many people passed.

#flashcards/algorithms/concept #flashcards/topic/complexity List the parts of a finite automaton.::A list of states (few, never growing), one start state, a rule for every pair (current state, next symbol) saying which state to move to, and some accepting states. At the end of the input it says yes if it is in an accepting state.

#flashcards/algorithms/concept #flashcards/topic/complexity Why is the memory of a finite automaton tiny and fixed?::It stores only the current state, so reading a million symbols uses the same tiny memory as reading ten. Each symbol costs one step.

#flashcards/algorithms/concept #flashcards/topic/complexity What can a finite automaton not do?::A fixed number of states means a fixed amount of memory. It cannot count without limit, so it cannot check that brackets match to any depth.

#flashcards/algorithms/formula #flashcards/topic/complexity Write the update rule of a finite automaton and explain each symbol.
?
$$q_{t+1}=\delta(q_t,\,x_t)$$
- $q_t$: the state before reading the $t$-th symbol. $q_{t+1}$: the state after. The machine starts at a chosen start state $q_0$.
- $x_t$: the $t$-th input symbol.
- $\delta$ (delta): the transition table. It is all the machine is. There is no other memory, so what the machine "remembers" is exactly what the states can distinguish.
- The input is accepted if the last state is an accepting state.

#flashcards/algorithms/example #flashcards/topic/complexity A machine tests whether a binary number is divisible by 3, with $\delta(q,b)=(2q+b)\bmod 3$. Read `110` (which is 6).::Start $q_0=0$. Read 1: $(2\cdot0+1)\bmod3=1$. Read 1: $(2\cdot1+1)\bmod3=0$. Read 0: $(0+0)\bmod3=0$. The last state is 0, an accepting state, so the answer is yes, and 6 is indeed divisible by 3. Three states (the remainders) are enough.

#flashcards/algorithms/image #flashcards/topic/complexity ![[turnstile-state-machine.svg]]
What does this picture show?
?
A finite state machine for a turnstile: two states, Locked and Unlocked (the filled dot marks the start), and two inputs, Coin and Push. A coin unlocks it, a push through the unlocked turnstile locks it again, and the other two arrows leave the state unchanged. (From the note's caption: Chetvorno, Wikimedia Commons, CC0. The note read the states and arrows off the picture.)
