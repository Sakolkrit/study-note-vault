---
tags: [vector, linear-algebra, math, embedding, LLM]
status: growing
level: 4
tier: 1
---

# Vector

## 🟢 1. Simple (age 12)
A vector is a **list of numbers** that can also be drawn as an **arrow**. The arrow has a direction and a length.

**Example:** "Walk 3 steps east and 4 steps north" is the vector `[3, 4]`. The arrow starts where you stand and ends where you arrive. Two friends with different arrows end up in different places.

## 🔵 2. How it works
- **Make one:** write numbers in a row: `[3, 4]`. Each number is a **component** (how far along one direction).
- **Add two vectors:** add matching numbers. `[3, 4] + [1, 2] = [4, 6]`. Drawing it: put the second arrow's tail at the first arrow's tip; the sum goes from the start to the end.
- **Stretch one:** multiply every number. `2 × [3, 4] = [6, 8]` is the same direction, twice as long.
- **Length:** Pythagoras. `[3, 4]` has length √(3² + 4²) = 5.
- **Not only 2 numbers:** a vector can have 3 numbers (3D space) or 768 numbers. You can't draw 768 dimensions, but the same rules still work.

In AI, a vector is an **address for meaning**: a word, sentence or picture becomes a long list of numbers ([[Embedding]]), and we compare lists with [[Cosine Similarity]].

![[vector-addition.png]]
*Adding two vectors: the red arrow is their sum. Image: KJBurns, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vector-parallelogram.PNG), CC BY-SA 3.0.*

## 🟠 3. Technical detail
A vector $v\in\mathbb{R}^n$ is an ordered $n$-tuple of reals. Operations:

| Operation | Rule | Meaning |
|---|---|---|
| Add | $(u+v)_i=u_i+v_i$ | combine movements |
| Scale | $(cv)_i=c\,v_i$ | change length (and flip if $c<0$) |
| Dot product | $u\cdot v=\sum_i u_iv_i$ | how aligned they are |
| Norm (length) | $\lVert v\rVert=\sqrt{v\cdot v}$ | size |
| Unit vector | $v/\lVert v\rVert$ | direction only |

**Where:**
- $u$ and $v$ are two vectors of the same length; $c$ is an ordinary number (a *scalar*).
- $i$ is the position in the list (1st number, 2nd number, ...). $u_i$ means "the $i$-th number of $u$". **Matters:** every rule works position by position.
- $\sum_i$ means "add up over all positions". **Matters:** in the dot product it turns a list of products into one single number.
- $\sqrt{v\cdot v}$ is Pythagoras: square every number, add, take the root. **Matters:** it gives the arrow's length, e.g. $\sqrt{3^2+4^2}=5$.
- $v/\lVert v\rVert$ divides every number by the length. **Matters:** the result has length exactly 1 and keeps only the direction, which is what [[Cosine Similarity]] compares.

Dot product also equals $\lVert u\rVert\,\lVert v\rVert\cos\theta$, which links it to the angle ([[Cosine Similarity]]). Here $\theta$ (theta) is the angle between the two arrows and $\cos\theta$ is 1 when they point the same way, 0 at a right angle, and -1 when opposite.

**Example (tiny numbers):** $u=[3,4]$, $v=[1,2]$.
- Add: $[3+1,\ 4+2]=\mathbf{[4,6]}$. Scale: $2u=\mathbf{[6,8]}$.
- Dot product: $3\cdot1+4\cdot2=\mathbf{11}$. Length of $u$: $\sqrt{9+16}=\mathbf{5}$.
- Unit vector of $u$: $[3/5,\ 4/5]=\mathbf{[0.6,\ 0.8]}$, which has length $\sqrt{0.36+0.64}=1$.
- Right angle check: $[1,0]\cdot[0,1]=0$ (east and north are fully independent directions).

Where vectors show up in AI:
- an [[Embedding]] is a vector (e.g. 768 numbers per token)
- model weights are matrices, i.e. stacks of vectors; a layer maps vectors to vectors
- training moves the weight vector downhill along the **gradient**, itself a vector
- search finds the stored vectors nearest to a query vector ([[Vector Database]])

```python
import numpy as np
u, v = np.array([3., 4.]), np.array([1., 2.])
print(u + v, 2 * u)                       # [4. 6.] [6. 8.]
print(np.linalg.norm(u), u @ v)           # 5.0 11.0
```

## 🔴 4. Deep / formal
A **vector space** over a field $F$ is a set $V$ with addition and scalar multiplication satisfying the 8 axioms (associativity, commutativity, zero, inverses, distributivity, ...). $\mathbb{R}^n$ is the standard example, but functions and matrices also form vector spaces.

- **Basis and dimension:** a basis is a linearly independent set that spans $V$; every vector is a unique combination $v=\sum_i c_ie_i$. The number of basis vectors is the dimension.
- **Inner product space:** an inner product $\langle u,v\rangle$ gives a norm $\lVert v\rVert=\sqrt{\langle v,v\rangle}$, angles, and orthogonality ($\langle u,v\rangle=0$). The **Cauchy–Schwarz inequality** $|\langle u,v\rangle|\le\lVert u\rVert\lVert v\rVert$ guarantees cosine stays in $[-1,1]$.
- **Norms:** a family of ways to measure length:

$$
\lVert v\rVert_p=\Big(\sum_i|v_i|^p\Big)^{1/p}
$$

**Where:**
- $|v_i|$ is the absolute value (size without sign) of the $i$-th number, so negatives do not cancel positives.
- $p$ is a chosen setting. $|v_i|^p$ raises each size to that power, and the outer $1/p$ power undoes it after adding. **Matters:** $p$ decides what counts as "long".
- $p=2$ is the normal straight-line (Euclidean) length. $p=1$ is Manhattan length (add the sizes, like walking city blocks). $p\to\infty$ is simply the biggest single number.
- In AI: $p=2$ is used to normalise embeddings; $p=1$ is used to encourage zeros in a vector (sparsity).

**Example (tiny numbers):** $v=[3,\,-4]$.
- $p=1$: $|3|+|-4|=\mathbf{7}$ (walk 3 blocks east, 4 blocks south).
- $p=2$: $\sqrt{9+16}=\mathbf{5}$ (the straight-line shortcut).
- $p\to\infty$: the biggest single size $=\mathbf{4}$.
- Same arrow, three different "lengths". Which one you use depends on what you want to measure.
- **Linear maps:** a matrix $A$ sends $v\mapsto Av$ and preserves addition and scaling. A neural layer is $v\mapsto\sigma(Wv+b)$: a linear map plus a nonlinearity.

**High-dimensional geometry:** in large $n$, random vectors are almost orthogonal (their cosine concentrates near 0 with spread about $1/\sqrt{n}$), and distances between random points concentrate. This is why embeddings can store many "directions" for different meanings in a modest number of dimensions, and why distance measures need care.

**What came before, and how this improved on it:**
- **Before:** describe a thing with separate named facts that you can only compare one at a time.
- **Its limit:** you cannot add, scale or measure "closeness" between whole things.
- **What changed:** write the facts as one ordered list of numbers that follows the rules of arithmetic. Vectors in this sense were developed in the 1800s (Hamilton, Grassmann, Gibbs; from memory, not checked). Their use for words became popular with [[Word2Vec 2013]].
- **How much better:** in the code below, four hand-made numbers per word, chosen by me so that it works, make `king - man + woman` equal the `queen` vector exactly, with a cosine similarity of 1.000 (the next closest word, king, scores 0.718). Real embeddings are only approximately like this. It is the arithmetic that is new, not the made-up numbers.

**Check it yourself:**

```python
import numpy as np
words = {"king": [0.9, 0.9, 0.1, 0.8], "queen": [0.9, 0.1, 0.9, 0.8], "man": [0.1, 0.9, 0.1, 0.1], "woman": [0.1, 0.1, 0.9, 0.1], "apple": [0.0, 0.5, 0.5, 0.0]}
v = {w: np.array(x) for w, x in words.items()}                     # 4 hand-made numbers per word: royal, male, female, person
q = v["king"] - v["man"] + v["woman"]
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print("king - man + woman, then the closest word by cosine similarity:")
for w in sorted(v, key=lambda w: -cos(q, v[w])): print(f"  {w:6s} {cos(q, v[w]):.3f}")
print("one subtraction and one addition on 4 numbers each; the same lines work for 4,000 numbers")
```

## Related
- [[Embedding]] · [[Cosine Similarity]] · [[Vector Database]] · [[Matrix]] · [[Gradient Descent]] · [[Eigenvalue]]

## Sources
- Strang, *Introduction to Linear Algebra*
- 3Blue1Brown, *Essence of Linear Algebra* (video series)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/linear-algebra Unit vector:::A vector divided by its own length. Its length is exactly 1, so it keeps only the direction.

#flashcards/math/concept #flashcards/topic/linear-algebra What is a vector, in plain words?::A list of numbers that can also be drawn as an arrow with a direction and a length.

#flashcards/math/example #flashcards/topic/linear-algebra What is the length of the vector [3, 4]?::5, from Pythagoras: √(3² + 4²).

#flashcards/math/example #flashcards/topic/linear-algebra What is the dot product of [3, 4] and [1, 2]?::11, because 3·1 + 4·2 = 11.

#flashcards/math/concept #flashcards/topic/linear-algebra In AI, what is an embedding?::A vector (a long list of numbers) that stands for the meaning of a word, sentence or picture.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the dot product of $u$ and $v$ with sum notation, and say what it tells you.
?
$$
u\cdot v=\sum_i u_iv_i
$$
One single number that says how aligned the two vectors are.
- $u$, $v$: two vectors of the same length.
- $i$: the position in the list (1st number, 2nd number, ...). $u_i$ is the $i$-th number of $u$.
- $\sum_i$: add up over all positions. It turns a list of products into one number.
- $u_iv_i$: multiply the two numbers at the same position.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the $p$-norm of a vector $v$, and name the cases $p=1$ and $p=2$.
?
$$
\lVert v\rVert_p=\Big(\sum_i|v_i|^p\Big)^{1/p}
$$
$p=1$ is Manhattan length (add the sizes). $p=2$ is the normal straight-line (Euclidean) length.
- $v_i$: the $i$-th number of the vector. $|v_i|$ is its size without the sign, so negatives do not cancel positives.
- $p$: a chosen setting that decides what counts as "long". Each size is raised to the power $p$.
- $\sum_i$: add all those powered sizes.
- $(\cdot)^{1/p}$: the outer power undoes the first one, so the result is a length again.

#flashcards/math/cloze #flashcards/topic/linear-algebra The dot product also equals $\lVert u\rVert\,\lVert v\rVert\cos\theta$. Here $\cos\theta$ is ==1== when the arrows point the same way, ==0== at a right angle and ==-1== when they point opposite ways.

#flashcards/math/cloze #flashcards/topic/linear-algebra For $v=[3,-4]$, the $p=1$ length is ==7==, the $p=2$ length is ==5== and the biggest single size ($p\to\infty$) is ==4==.

#flashcards/math/image #flashcards/topic/linear-algebra ![[vector-addition.png]]
In this picture, what does the red arrow show?
?
The sum of the two vectors: put the second arrow's tail at the first arrow's tip, and the sum goes from the start to the end.
