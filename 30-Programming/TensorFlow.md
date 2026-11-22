---
tags: [tensorflow, keras, python, programming, deep-learning, machine-learning, autograd, tensors]
status: growing
level: 4
tier: 2
---

# TensorFlow

TensorFlow is a [[Python]] library from Google for building and training [[Neural Network]]s, just as [[PyTorch]] is. It gives you the same two big things: **tensors** (grids of numbers like [[NumPy]] arrays, able to run on a [[GPU]]) and **automatic differentiation** (it works out the slopes needed for learning by itself). On top of it sits **Keras**, a friendly layer for stacking neural-network building blocks and training them in a few lines. If you know PyTorch, most ideas carry over, and this note points out where TensorFlow does things differently.

## 🟢 1. Simple (age 12)
Imagine a giant calculator that can do sums on whole tables of numbers at once, and that also remembers every step it did. When the answer is wrong, it can walk back through its steps and tell you "if you nudge this number a little, the answer gets better". That is what TensorFlow does. Because it remembers the steps, a computer can teach itself by making tiny nudges to millions of numbers until its answers are good.

**Example:** You show a computer 100 dots that roughly lie along a slanted line. TensorFlow starts with a bad guess for the slant, measures how far off it is, nudges the guess, and repeats. After a few rounds the guess is almost exactly right.

## 🔵 2. How it works
1. **Tensors hold the numbers.** A *tensor* is a table of numbers with any number of directions: a single number, a list, a grid, a stack of grids. A plain TensorFlow tensor **cannot be changed** after it is made. If you need numbers that learning can change, you use a `tf.Variable`.
2. **A tape records the maths.** When you wrap some calculations in a `tf.GradientTape()`, TensorFlow records each step on a "tape". Afterwards you ask the tape for the **gradient** (the slope) of the answer with respect to any variable. This is automatic differentiation (see [[Backpropagation]]).
3. **Take a step downhill.** An *optimizer* uses the gradient to nudge each variable a little in the direction that makes the error smaller (see [[Gradient Descent]]).
4. **Keras packs this up.** With Keras you list layers, say which optimizer and [[Loss Function]] to use, and call `fit`. Keras does the tape, the gradient and the nudging for you.
5. **Eager first, graph if you want speed.** TensorFlow 2 runs each line immediately as you write it (*eager* mode). If you wrap a function in `@tf.function`, TensorFlow traces it once into a **graph** of operations, which it can then optimise and run fast.

## 🟠 3. Technical detail
- **Versions in this note:** the code was run on TensorFlow 2.9.3 with Keras 2.9.0, Python 3.9, on a CPU only (no GPU was available here, so nothing about GPU speed or setup was tested). Names and behaviour can differ in other versions.
- **Tensors:** they have a `shape` and a `dtype`. Decimal numbers default to `float32` and whole numbers to `int32` (PyTorch uses `int64` for whole numbers). Tensors are read-only: the code shows that assigning into one raises a `TypeError`. A `tf.Variable` can be changed with `assign`. `tf.constant(numpy_array)` made a **copy** in the code: changing the NumPy array afterwards did not change the tensor. (PyTorch's `from_numpy` shares memory, so it differs here.)
- **`tf.GradientTape`:** record inside a `with` block, then call `tape.gradient(loss, variables)`. The code gets $d/dx(x^2+2x)=8$ at $x=3$ and the same gradients as the [[PyTorch]] note for its small network (3.0 and 12.0).
- **No build-up of gradients:** in PyTorch, gradients are *added* to the stored value on each `backward()` unless you reset them. In TensorFlow, each `tape.gradient(...)` call just returns a value, so there is nothing to reset. The code shows 5.0 on every call, where the PyTorch note shows 5, 10, 15. A normal tape can only be used once: a second `tape.gradient` call raised a `RuntimeError` (a `persistent=True` tape allows more).
- **Keras models:** `tf.keras.Sequential` stacks layers. `model.compile(optimizer, loss)` then `model.fit(X, Y, epochs, batch_size)` does the training loop. The code fits $y=3x+1$ plus noise and learns a weight of 2.98 and a bias of 1.01 after 100 full-batch steps (the PyTorch note gets 3.0 and 1.0 after 200 steps, so these two numbers are not directly comparable).
- **Parameters:** a `Dense` layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs has $(n_{\text{in}}+1)\,n_{\text{out}}$ parameters (equation 2 below). The code's network has 67, the same as the PyTorch note's.
- **Dropout:** it is only active when the model is called with `training=True` (see [[Dropout]]). The code shows two calls that differ with `training=True` and agree with `training=False`. `fit` sets this for you.
- **`tf.function`:** the first call *traces* the Python function into a graph. The Python code inside runs only while tracing, so a `print` or a list append inside it runs once, not on every call. In the code, three calls with the same shape and type ran the Python body once. A new shape and a new data type each caused another trace (3 traces in all). The results match eager results. Retracing a lot is a common slowdown, but I did not time anything here.
- **Batches:** each step of training uses one batch of the data, so one epoch takes $\lceil N/B\rceil$ steps (equation 3 below). The code counts 4 steps for 100 samples with a batch size of 32.
- **TensorFlow against PyTorch** (from memory about the wider ecosystem; I did not check the current state of either): TensorFlow has tools for putting models on phones and servers (TensorFlow Lite, TensorFlow Serving) and for watching training (TensorBoard). PyTorch is more common in research code. Many recent models are written in PyTorch, which is why most snippets in this vault use it. I did not run any of the TensorFlow tools named here.
- **Limits:** the code runs tiny examples on a CPU, and checks maths and counts. It does not time anything, use a GPU, save or load a model, or use the data-loading tools (`tf.data`).

The code covers seven things: (1) tensors, read-only against Variables, (2) the tape and its gradients, (3) one gradient step by hand and with an optimizer, (4) a Keras fit of a line, (5) a small network's parameter count and Dropout, (6) `tf.function` tracing, (7) steps per epoch.

```python
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, tensorflow as tf
tf.random.set_seed(0); np.random.seed(0)
# 1) tensors: shape, dtype, maths, broadcasting; a plain tensor cannot be changed, a Variable can
a = tf.constant([[1., 2.], [3., 4.]])
print(a.shape, a.dtype.name, '|', (a @ a).numpy().tolist(), '|', (a + tf.constant([100., 200.])).numpy().tolist())
n = np.zeros(3); t = tf.constant(n); n += 1; print('numpy array changed later; the tensor kept its copy:', t.numpy().tolist(), '| int default:', tf.constant([1, 2]).dtype.name)
try: a[0, 0] = 9.0
except TypeError: print('assigning into a tensor -> TypeError (tensors are read-only)')
v = tf.Variable([1., 2.]); v.assign([5., 6.]); v[0].assign(7.); print('a Variable can change:', v.numpy().tolist())
# 2) GradientTape: derivatives of whatever happened inside the "with" block
x = tf.Variable(3.0)
with tf.GradientTape() as tape: y = x ** 2 + 2 * x
print('d/dx (x^2+2x) at 3 =', tape.gradient(y, x).numpy())
w1, w2 = tf.Variable(0.5), tf.Variable(2.0)
with tf.GradientTape() as tape:
    loss = (w2 * tf.nn.relu(w1 * 3.0) - 2.0) ** 2
g1, g2 = tape.gradient(loss, [w1, w2]); print('loss', loss.numpy(), '| dL/dw2', g2.numpy(), '| dL/dw1', g1.numpy(), '(by hand: 3.0 and 12.0)')
w = tf.Variable(1.0)
for k in (1, 2, 3):
    with tf.GradientTape() as tp: z = w * 5
    print('call', k, ': gradient =', tp.gradient(z, w).numpy(), '(a fresh tape each time: no build-up)')
with tf.GradientTape() as tp: z = w * 5
tp.gradient(z, w)
try: tp.gradient(z, w)
except RuntimeError: print('using a normal tape a second time -> RuntimeError')
# 3) one gradient step by hand, then with an optimizer
xs, ys = tf.constant([1., 2., 3.]), tf.constant([2., 4., 6.]); w = tf.Variable(0.0)
with tf.GradientTape() as tape: L = tf.reduce_mean((w * xs - ys) ** 2)
g = tape.gradient(L, w); print('loss', round(float(L), 3), '| gradient', round(float(g), 3), '| new w', round(0.0 - 0.05 * float(g), 4))
tf.keras.optimizers.SGD(0.05).apply_gradients([(g, w)]); print('optimizer gives the same w:', round(float(w), 4))
# 4) Keras: fit y = 3x + 1 with noise
X = np.linspace(-1, 1, 100).astype('float32').reshape(-1, 1); Y = 3 * X + 1 + 0.1 * np.random.randn(100, 1).astype('float32')
m = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))]); m.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='mse')
h = m.fit(X, Y, epochs=100, batch_size=100, verbose=0)
print('learned weight and bias:', round(float(m.layers[0].kernel[0, 0]), 2), round(float(m.layers[0].bias[0]), 2), '| final loss', round(h.history['loss'][-1], 4))
# 5) a small network: count parameters; Dropout is only active when training=True
net = tf.keras.Sequential([tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)), tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(3)])
print('parameters:', net.count_params(), '(= (4+1)*8 + (8+1)*3)')
inp = tf.ones((1, 4)); o1, o2 = net(inp, training=True), net(inp, training=True); e1, e2 = net(inp, training=False), net(inp, training=False)
print('training=True: two calls differ:', not np.allclose(o1, o2), '| training=False: two calls equal:', np.allclose(e1, e2))
# 6) tf.function: turn Python into a graph; the Python body runs only while tracing
traces = []
@tf.function
def f(x):
    traces.append(1); return x * 2 + 1
f(tf.constant(1.0)); f(tf.constant(2.0)); f(tf.constant(3.0)); print('same shape and type, 3 calls: Python body ran', len(traces), 'time(s)')
f(tf.constant([1.0, 2.0])); f(tf.constant(1)); print('after a new shape and a new dtype:', len(traces), 'traces | result', f(tf.constant(4.0)).numpy(), '| eager gives', 4.0 * 2 + 1)
# 7) batches: one step per batch, so steps per epoch = ceil(N / B)
class Count(tf.keras.callbacks.Callback):
    steps = 0
    def on_train_batch_end(self, b, logs=None): Count.steps += 1
m.fit(X, Y, epochs=1, batch_size=32, verbose=0, callbacks=[Count()]); print('100 samples, batch 32: steps in one epoch =', Count.steps, '| ceil(100/32) =', -(-100 // 32))
# (2, 2) float32 | [[7.0, 10.0], [15.0, 22.0]] | [[101.0, 202.0], [103.0, 204.0]]
# numpy array changed later; the tensor kept its copy: [0.0, 0.0, 0.0] | int default: int32
# assigning into a tensor -> TypeError (tensors are read-only)
# a Variable can change: [7.0, 6.0]
# d/dx (x^2+2x) at 3 = 8.0
# loss 1.0 | dL/dw2 3.0 | dL/dw1 12.0 (by hand: 3.0 and 12.0)
# call 1 : gradient = 5.0 (a fresh tape each time: no build-up)
# call 2 : gradient = 5.0 (a fresh tape each time: no build-up)
# call 3 : gradient = 5.0 (a fresh tape each time: no build-up)
# using a normal tape a second time -> RuntimeError
# loss 18.667 | gradient -18.667 | new w 0.9333
# optimizer gives the same w: 0.9333
# learned weight and bias: 2.98 1.01 | final loss 0.0101
# parameters: 67 (= (4+1)*8 + (8+1)*3)
# training=True: two calls differ: True | training=False: two calls equal: True
# same shape and type, 3 calls: Python body ran 1 time(s)
# after a new shape and a new dtype: 3 traces | result 9.0 | eager gives 9.0
# 100 samples, batch 32: steps in one epoch = 4 | ceil(100/32) = 4
```
Reading the output (TensorFlow 2.9.3 on a CPU, Python 3.9):
- **Tensors:** a 2×2 `float32` tensor, with the matrix product `[[7, 10], [15, 22]]`, and the row `[100, 200]` stretched over both rows. The tensor kept its zeros when the NumPy array changed afterwards, and whole numbers default to `int32`. Assigning into a tensor raises `TypeError`, and a Variable changed to `[7.0, 6.0]`.
- **Tape:** the slope is 8.0. The small network gives a loss of 1.0, with $dL/dw_2=3.0$ and $dL/dw_1=12.0$, which are the hand-worked values. Three calls each give 5.0. A second use of the same normal tape raises `RuntimeError`.
- **Gradient step:** the loss is 18.667, the gradient is −18.667 and the new weight is 0.9333, the same as the PyTorch note's numbers, and the optimizer gives 0.9333 too.
- **Keras:** the learned weight and bias are 2.98 and 1.01 and the final loss is 0.0101.
- **Network:** 67 parameters. With `training=True` two calls differ, and with `training=False` they are equal.
- **`tf.function`:** the Python body ran once for 3 calls, and 3 times in total after a new shape and a new dtype. The result 9.0 equals the eager value.
- **Batches:** 4 steps for 100 samples with a batch size of 32.
- **Limits:** CPU only, small inputs, nothing timed.

## 🔴 4. Deep / formal

**One gradient step.** Fit a line through the origin, $\hat y=wx$, to data $(x_i,y_i)$ by minimising the mean squared error. The tape computes the slope of $L$ with respect to $w$, and the optimizer steps against it:

$$L(w)=\frac1n\sum_{i=1}^{n}(w x_i-y_i)^2,\qquad \frac{\partial L}{\partial w}=\frac2n\sum_{i=1}^{n}x_i\,(w x_i-y_i),\qquad w\leftarrow w-\eta\,\frac{\partial L}{\partial w}$$

**Where:**
- $w$ is the single number being learned (a `tf.Variable`). **Matters:** it is the thing the tape differentiates with respect to. A plain tensor would not be tracked, so no gradient would be found.
- $x_i,y_i$ are the inputs and targets, and $n$ is how many there are. **Matters:** dividing by $n$ makes the loss an average, so the size of the step does not change when the number of data points does.
- $(wx_i-y_i)$ is the error on point $i$. **Matters:** its square makes big errors count much more, and the factor 2 in the derivative comes from differentiating the square.
- $\partial L/\partial w$ is the slope of the loss at the current $w$. **Matters:** it points uphill, so the step goes the opposite way.
- $\eta$ is the learning rate (0.05 here; see [[Learning Rate]]). **Matters:** too big and the step overshoots, too small and learning crawls.

**Example (tiny numbers):** $x=(1,2,3)$, $y=(2,4,6)$ (so the true rule is $y=2x$), start at $w=0$, $\eta=0.05$.
1. Errors at $w=0$: $0-2=-2$, $0-4=-4$, $0-6=-6$. Loss: $(4+16+36)/3=18.667$.
2. Slope: $\frac23\big(1\cdot(-2)+2\cdot(-4)+3\cdot(-6)\big)=\frac23(-28)=-18.667$.
3. Step: $w=0-0.05\cdot(-18.667)=0.9333$.
4. The code's tape gave a loss of 18.667, a gradient of −18.667 and a new weight of 0.9333, and the optimizer's step gave the same 0.9333. These are the same numbers as in the [[PyTorch]] note, so both libraries agree.
5. Meaning: the slope is negative, so raising $w$ lowers the loss. One step moves $w$ from 0 to about 0.93, on the way to the true value 2.

**How many parameters?** A fully connected layer (`Dense`) multiplies its inputs by a weight matrix and adds a bias for each output. With $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs it holds:

$$\#\text{params}=(n_{\text{in}}+1)\,n_{\text{out}}$$

**Where:**
- $n_{\text{in}}$ is the number of numbers going into the layer. **Matters:** each output has one weight for every input, so the weights number $n_{\text{in}}\times n_{\text{out}}$.
- $n_{\text{out}}$ is the number of units (outputs) in the layer. **Matters:** each unit has its own set of weights and its own bias.
- The $+1$ is the bias for each output. **Matters:** the bias lets a unit shift its answer up or down, so it can fit lines that do not pass through the origin.
- Layers with no learned numbers (Dropout, ReLU) add 0. **Matters:** the total is the sum over the Dense layers only.

**Example (tiny numbers):** the code's network, 4 inputs → Dense(8) → Dropout → Dense(3).
1. First layer: $(4+1)\times8=40$.
2. Second layer: $(8+1)\times3=27$.
3. Total: $40+27=67$ (Dropout adds nothing). The code counted 67.
4. Meaning: even this tiny network has 67 numbers for the optimizer to adjust, and this count grows with the *product* of the layer widths, so wide layers get expensive quickly. The PyTorch note's network is the same shape and has the same 67.

**Steps in an epoch.** Training goes through the data in batches of size $B$. One pass over all $N$ samples is an *epoch*, and it takes this many steps (each step is one gradient update):

$$\text{steps per epoch}=\left\lceil\frac{N}{B}\right\rceil$$

**Where:**
- $N$ is the number of training samples. **Matters:** more data means more steps in each epoch.
- $B$ is the batch size (how many samples are used for one gradient step). **Matters:** a small $B$ gives many noisy steps, and a large $B$ gives few smooth but heavier steps.
- $\lceil\cdot\rceil$ means "round up". **Matters:** the last batch is smaller if $N$ is not a multiple of $B$, and it still counts as a step.
- The total number of steps is epochs × steps per epoch. **Matters:** it is what the optimizer actually runs, and the number to compare when you change $B$.

**Example (tiny numbers):** $N=100$ samples, $B=32$.
1. $100/32=3.125$, which rounds up to 4.
2. Batches: 32, 32, 32 and a last batch of 4 (since $3\times32=96$, and $100-96=4$).
3. The code counted 4 steps with a callback that counts batches.
4. Meaning: the last small batch still gets its own step, so with $B=32$ one epoch gives 4 gradient steps, not 3.

**Other points (from memory, not checked here):**
- TensorFlow was released by Google in 2015, and TensorFlow 2 in 2019 made eager execution the default. I did not check either date.
- Keras was a separate library by François Chollet before it was bundled into TensorFlow as `tf.keras`, and recent Keras versions (Keras 3) can run on other back ends. The code used the version that came with TensorFlow 2.9, and I did not check the current situation.
- That native Windows GPU support stopped after TensorFlow 2.10 (so recent versions need WSL2 for a GPU) is from memory; I could not test it.
- TensorFlow Lite, TensorFlow Serving, TensorBoard and `tf.data` exist and are widely used. I did not run any of them.
- JAX is another library that uses a similar tracing idea. I did not cover or install it.

**What came before, and how this improved on it:**
- **Before:** write the network and its gradients by hand, or use research libraries such as Theano that built graphs for a single machine.
- **Its limit:** hand-written gradients are slow to write, and code tied to Python is hard to optimise and deploy.
- **What changed:** describe the computation as a graph that can be differentiated, optimised and run on many devices. Google released TensorFlow in November 2015 (from memory, not checked).
- **How much better:** in the code below (TensorFlow 2.9.3), the gradient of $x^3+2x$ at 2 by tape is 14.0, the same as the hand value $3\cdot 2^2+2$. And `tf.function` traced a Python function into a graph of 7 operations that can run without Python. These show the mechanism and not speed; I measured no speedup.

**Check it yourself:**

```python
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
x = tf.Variable(2.0)
with tf.GradientTape() as tape: y = x ** 3 + 2 * x
print("TensorFlow", tf.__version__, " gradient of x^3 + 2x at 2 by tape:", float(tape.gradient(y, x)), " by hand 3 * 2^2 + 2 =", 3 * 4 + 2)
@tf.function
def f(a): return tf.reduce_sum(a * a) + 1.0
g = f.get_concrete_function(tf.TensorSpec([None], tf.float32)).graph
print("the same python function traced once into a graph of", len(g.get_operations()), "operations, which can then be optimised and run without Python:")
print(sorted({o.type for o in g.get_operations()}))
```

## Related
[[PyTorch]] · [[NumPy]] · [[Python]] · [[Neural Network]] · [[Backpropagation]] · [[Gradient Descent]] · [[Loss Function]] · [[Learning Rate]] · [[Dropout]] · [[Stochastic Gradient Descent]] · [[GPU]] · [[Matplotlib]] · [[Pandas]] · [[Command Line]] · [[Virtual Environments]]
- [[FastAPI]] · [[Docker]]

## Sources
- TensorFlow documentation (tensorflow.org), on `tf.GradientTape`, `tf.function` and Keras (from memory; I did not open the pages for this note).
- Python and library versions: TensorFlow 2.9.3, Keras 2.9.0, NumPy 1.24.4, Python 3.9.6, checked by importing them.
- No figure: Wikimedia Commons has only logos and an unrelated neural-network illustration for TensorFlow. The logo is a trademark and not explanatory, so I did not use it.
- The code and every Example number were run for this note (TensorFlow 2.9.3, CPU). Operations were counted, not timed.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dl-frameworks TensorFlow:::A giant calculator that does sums on whole tables of numbers at once and remembers every step, so it can walk back and tell you which way to nudge each number.

#flashcards/programming/concept #flashcards/topic/dl-frameworks Why is `tf.Variable` needed?::A plain TensorFlow tensor cannot be changed after it is made. For numbers that learning can change, use a `tf.Variable`.

#flashcards/programming/concept #flashcards/topic/dl-frameworks What does `tf.GradientTape()` do?::It records each calculation on a tape. Afterwards you ask the tape for the gradient of the answer with respect to any variable (automatic differentiation).

#flashcards/programming/concept #flashcards/topic/dl-frameworks What does Keras do for you?::You list layers, name the optimizer and loss, and call `fit`. Keras does the tape, the gradient and the nudging.

#flashcards/programming/concept #flashcards/topic/dl-frameworks Eager mode against `@tf.function`?::TensorFlow 2 runs each line immediately (eager). Wrapping a function in `@tf.function` traces it once into a graph of operations that can be optimised and run fast.

#flashcards/programming/formula #flashcards/topic/dl-frameworks Write the loss, slope and update for one learned weight and explain each symbol.
?
$$L(w)=\frac1n\sum_{i=1}^{n}(w x_i-y_i)^2,\qquad \frac{\partial L}{\partial w}=\frac2n\sum_{i=1}^{n}x_i\,(w x_i-y_i),\qquad w\leftarrow w-\eta\,\frac{\partial L}{\partial w}$$
- $w$: the single number being learned (a `tf.Variable`), the thing the tape differentiates. A plain tensor would not be tracked, so no gradient would be found.
- $x_i,y_i$: inputs and targets. $n$: how many. Dividing by $n$ makes the loss an average, so the step size does not change with the number of data points.
- $(wx_i-y_i)$: the error on point $i$. Squaring makes big errors count more.
- $\eta$: the learning rate.

#flashcards/programming/example #flashcards/topic/dl-frameworks $x=(1,2,3)$, $y=(2,4,6)$, $w=0$, $\eta=0.05$. What are the loss, gradient and new $w$?::Errors $-2,-4,-6$. Loss $(4+16+36)/3=18.667$. Slope $\frac23(-28)=-18.667$. New $w=0-0.05\cdot(-18.667)=0.9333$. The note's tape and the optimizer's step both gave 0.9333.

#flashcards/programming/formula #flashcards/topic/dl-frameworks Write the number of parameters of a dense layer and explain each symbol.
?
$$\#\text{params}=(n_{\text{in}}+1)\,n_{\text{out}}$$
- $n_{\text{in}}$: the number of numbers going into the layer. Each output has one weight for every input, so $n_{\text{in}}\times n_{\text{out}}$ weights.
- $n_{\text{out}}$: the number of units (outputs). Each unit has its own weights and its own bias.
- The $+1$: the bias for each output, so a unit can shift its answer and fit lines that do not pass through the origin.

#flashcards/programming/example #flashcards/topic/dl-frameworks A network 4 inputs → Dense(8) → Dropout → Dense(3). How many parameters?::First layer $(4+1)\times8=40$. Second $(8+1)\times3=27$. Total 67 (Dropout adds nothing). The code counted 67. The count grows with the product of the layer widths, so wide layers get expensive quickly.

#flashcards/programming/formula #flashcards/topic/dl-frameworks Write the number of gradient steps per epoch and explain each symbol.
?
$$\text{steps per epoch}=\left\lceil\frac{N}{B}\right\rceil$$
- $N$: the number of training samples. More data means more steps per epoch.
- $B$: the batch size, how many samples are used for one gradient step. A small $B$ gives many noisy steps, a large $B$ few smooth but heavier steps.
- $\lceil\cdot\rceil$: round up. The last batch is smaller if $N$ is not a multiple of $B$, and it still counts as a step.
- Total steps are epochs times steps per epoch.

#flashcards/programming/example #flashcards/topic/dl-frameworks $N=100$ samples, $B=32$. How many steps per epoch?::$100/32=3.125$, rounded up to 4. Batches of 32, 32, 32 and a last batch of 4 ($100-96=4$). The code counted 4 steps with a batch-counting callback.
