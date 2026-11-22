---
tags: [programming, python, NumPy, machine-learning, performance, linear-algebra]
status: growing
level: 4
tier: 1
---

# Vectorization

## 🟢 1. Simple (age 12)
Vectorization means **giving the computer the whole list at once instead of one number at a time**. If you must add 1,000,000 prices, you can either hand them over one by one and wait for each answer, or hand over the whole pile and say "do them all". The second way is much faster, and your instructions are shorter too.

**Example:** A teacher needs to give every student in a class a sticker. One way: walk to each desk, hand a sticker, write a tick, walk on. The other way: hand each row a stack of stickers and say "pass them along". Same result, far less walking.

## 🔵 2. How it works
1. Many [[Machine Learning]] formulas are a sum of products. For example, a house price from many features is $w_1x_1+w_2x_2+\dots+b$ (see [[Linear Regression]]).
2. The slow way is a **loop**: take $w_1x_1$, add it to a running total, take $w_2x_2$, and so on, one Python step at a time.
3. The fast way is a single call such as `np.dot(w, x)` in [[NumPy]]. The numbers sit side by side in memory, and a fast library written in a lower-level language runs the whole loop.
4. Modern computer chips can also multiply several numbers at the same moment, and a [[GPU]] can do thousands. A loop in Python cannot use that, but one vectorized call can.
5. The answer is the same. The code is shorter and easier to read, and it runs much faster.

## 🟠 3. Technical detail
- **Notation:** a [[Vector]] $\vec w$ holds all the weights, $\vec x$ all the features. The dot product $\vec w\cdot\vec x$ is the vectorized form of the sum.
- **Why it is quick:**
  - Python loops pay a cost for every step (looking up types, handling each number as an object). A NumPy array stores raw numbers in one block and a compiled routine loops over it.
  - The routine can use SIMD instructions (one instruction on several numbers) and, for matrices, tuned linear-algebra libraries. A [[GPU]] pushes the same idea much further.
- **Beyond one example:** with a matrix $X$ (one row per example) you compute all predictions at once with `X @ w + b`, and the gradient in [[Gradient Descent]] as `X.T @ error / m`. No loop over examples or features.
- **Rule of thumb:** if you wrote `for` over numbers in a list inside a training loop, look for a NumPy or [[PyTorch]] operation that does it in one go.
- **Cost:** it needs the data in an array, and it creates temporary arrays (memory). Very clear loops are still fine for small data.

```python
import numpy as np
w = np.array([1.0, 2.0, 3.0]); x = np.array([4.0, 5.0, 6.0]); b = 1.0
print(np.dot(w, x) + b)                       # 33.0, no loop
f = 0.0
for j in range(3):                            # the loop version of the same thing
    f = f + w[j] * x[j]
print(f + b)                                  # 33.0
X = np.array([[4.0, 5.0, 6.0], [1.0, 0.0, 2.0]])
print(X @ w + b)                              # [33.  8.]  two predictions in one line
```

## 🔴 4. Deep / formal
**One sum, two ways to compute it.** The prediction of a linear model with $n$ features is:

$$
f_{\vec w,b}(\vec x)=\vec w\cdot\vec x+b=\sum_{j=1}^{n} w_j x_j + b
$$

**Where:**
- $\vec w=(w_1,\dots,w_n)$ are the weights (how much each feature counts). $\vec x=(x_1,\dots,x_n)$ is one example's features. $b$ is the bias, a number added at the end.
- $\vec w\cdot\vec x$ is the **dot product**: multiply matching entries, then add up.
- $n$ is the number of features. **Matters:** the sum needs $n$ multiplications and $n-1$ additions either way. Vectorization does not change that count; it changes **who does the work and how**: a compiled routine on a block of memory instead of the Python interpreter, one number at a time.

**Example (tiny numbers):** $\vec w=(1,2,3)$, $\vec x=(4,5,6)$, $b=1$.
- Products: $1\cdot4=4$, $2\cdot5=10$, $3\cdot6=18$.
- Sum of products: $4+10+18=32$. Add $b$: $32+1=\mathbf{33}$.
- Meaning: the loop and `np.dot` both give 33. They differ only in how long they take once $n$ is large.

**Limits and trade-offs:**
- **Same arithmetic, not a smarter algorithm.** It does not reduce the amount of work. For example, a vectorized sort is still the same $O(n\log n)$ (see [[Big-O]]).
- **Memory:** vectorizing a huge computation may need huge temporary arrays. Splitting into batches (as in [[Stochastic Gradient Descent]]) fixes that.
- **Floating point:** the order of additions can differ between a loop and a library routine, so results can differ in the last digits (the check below uses a tolerance).

**What came before, and how this improved on it:**
- **Before:** write every sum as an explicit loop in the program.
- **Its limit:** in Python each loop step is slow, so a model with a million features or a million examples takes a very long time per update.
- **What changed:** hand the whole array to one library call ([[NumPy]]), which runs a compiled loop and can use the chip's parallel instructions.
- **How much better:** in the code below (a dot product of 1,000,000 numbers, best of 5 runs on my machine) the Python-list loop took about 0.07 to 0.09 s and `np.dot` about 0.0005 s, which was 136 times faster in one run and 159 times in another, with the same answer. The exact times change from run to run and machine to machine; the large gap does not.

**Check it yourself:**

```python
import time, numpy as np
n = 1_000_000
rng = np.random.default_rng(0)
w = rng.random(n); x = rng.random(n); b = 1.0
wl, xl = w.tolist(), x.tolist()
def pyloop():
    f = 0.0
    for j in range(n):
        f = f + wl[j] * xl[j]
    return f + b
def best(fn, r=5):
    ts = []
    for _ in range(r):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return min(ts), out
t1, o1 = best(pyloop)
t2, o2 = best(lambda: np.dot(w, x) + b)
print("loop %.4fs | np.dot %.5fs | ratio %.0fx | same answer %s" % (t1, t2, t1 / t2, abs(o1 - o2) < 1e-6 * abs(o2)))
```

## Related
- [[NumPy]] · [[Vector]] · [[Matrix]] · [[Linear Regression]] · [[Gradient Descent]] · [[GPU]] · [[PyTorch]] · [[Big-O]] · [[Machine Learning]] · [[Stochastic Gradient Descent]] · [[Machine Learning Specialization Course 1 Week 2]]

## Sources
- NumPy documentation (array operations and broadcasting)
- Course videos "Machine Learning Specialization" (Andrew Ng), Course 1 Week 2 (transcripts read; see [[Machine Learning Specialization]])
- The 136x to 159x figures are measured in this note's code (two runs), not taken from a source

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/data-libs Vectorization:::Giving the computer the whole list at once instead of one number at a time. The code is shorter and runs much faster.

#flashcards/programming/concept #flashcards/topic/data-libs Why is `np.dot(w, x)` faster than a Python loop?::Python loops pay a cost for every step (looking up types, handling each number as an object). A NumPy array stores raw numbers in one block and a compiled routine loops over it, possibly with SIMD instructions (one instruction on several numbers). A GPU pushes the same idea much further.

#flashcards/programming/formula #flashcards/topic/data-libs Write the linear model as a dot product and explain each symbol.
?
$$f_{\vec w,b}(\vec x)=\vec w\cdot\vec x+b=\sum_{j=1}^{n} w_j x_j + b$$
- $\vec w=(w_1,\dots,w_n)$: the weights (how much each feature counts). $\vec x=(x_1,\dots,x_n)$: one example's features. $b$: the bias, added at the end.
- $\vec w\cdot\vec x$: the dot product, multiply matching entries then add up.
- $n$: the number of features. The sum needs $n$ multiplications and $n-1$ additions either way. Vectorization does not change that count. It changes who does the work and how.

#flashcards/programming/example #flashcards/topic/data-libs $\vec w=(1,2,3)$, $\vec x=(4,5,6)$, $b=1$. What is $f$?::Products $4,10,18$, sum $32$, plus $b$ gives $33$. The loop and `np.dot` both give 33. They differ only in how long they take once $n$ is large.

#flashcards/programming/concept #flashcards/topic/data-libs How do you compute all predictions and the gradient for a whole matrix of examples?::With a matrix $X$ (one row per example): predictions as `X @ w + b`, and the gradient in gradient descent as `X.T @ error / m`. No loop over examples or features.

#flashcards/programming/concept #flashcards/topic/data-libs What is the rule of thumb for loops in a training loop?::If you wrote `for` over numbers in a list inside a training loop, look for a NumPy or PyTorch operation that does it in one go.

#flashcards/programming/concept #flashcards/topic/data-libs What does vectorization cost?::It needs the data in an array and creates temporary arrays (memory). Very clear loops are still fine for small data.
