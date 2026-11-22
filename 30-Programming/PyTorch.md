---
tags: [pytorch, python, programming, deep-learning, machine-learning, autograd, tensors]
status: growing
level: 4
tier: 2
---

# PyTorch

PyTorch is a [[Python]] library for building and training [[Neural Network]]s. It gives you two big things: **tensors** (grids of numbers like [[NumPy]] arrays, but able to run on a [[GPU]]) and **automatic differentiation** (it works out the slopes needed for learning by itself). Many notes in this vault use PyTorch in their code snippets, so this note explains the parts they rely on.

## 🟢 1. Simple (age 12)
Imagine you are tuning a radio with many knobs, to get the clearest sound. You turn a knob a little, listen, and decide whether to turn it more or the other way. Doing this for thousands of knobs by hand would be impossible. PyTorch is a machine that **remembers every knob you touched, listens to how wrong the sound is, and tells you exactly which way and how much to turn each knob**. You just repeat: check, ask which way to turn, turn a little.

**Example:** A robot throws a ball at a target and misses by 2 metres. PyTorch's job is to tell the robot, "if you throw 1 degree higher, you miss by 1.5 metres; if you throw with more power, you miss by 3 metres". With that information, the robot knows which knob to turn and by how much.

## 🔵 2. How it works
1. **Tensors are the numbers.** A tensor is a grid of numbers: a single number, a list, a table, or a stack of tables. Each has a **shape** (its size in each direction) and a **dtype** (what kind of numbers: decimals or whole numbers).
2. **Do maths on whole grids,** just as in NumPy: add, multiply, and the matrix product (see [[Matrix]]).
3. **Mark the numbers you want to learn.** Weights of a network are tensors that PyTorch is told to watch (`requires_grad`).
4. **Run the model forward.** As you compute (multiply, add, and so on), PyTorch quietly **records each step** in a chain, like a trail of breadcrumbs. The chain is called the computation graph.
5. **Measure how wrong it is** with a single number, the **loss** (see [[Loss Function]]).
6. **Ask for the slopes.** Calling `loss.backward()` walks the breadcrumb trail *backwards* and works out, for every watched number, how much the loss changes if that number changes a little. This is [[Backpropagation]], done for you.
7. **Take a small step** against the slope, using an optimiser such as [[Stochastic Gradient Descent]] or [[Adam Optimizer]]. The loss gets a bit smaller.
8. **Repeat** thousands of times. That repeated cycle is the **training loop**.

![[pytorch-computation-graph.svg]]
*A computation graph in general: the arrows show which results feed into which. Here the two multiplications (x = 2×5 and y = 5×7) both feed the final addition x + y. PyTorch records such a graph as you compute, then walks it backwards to get the slopes. The diagram is a general graph of operations, with "source" and "sink" end points, and is not drawn by PyTorch itself. Image: BrentPrinciple, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Computation_graph_example.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Tensors:** `torch.tensor([[1., 2.], [3., 4.]])`, `torch.zeros(2, 3)`, `torch.randn(5)`. Useful fields: `.shape`, `.dtype`, `.device`. Decimals default to 32-bit floats, and whole numbers to 64-bit integers (the code shows both).
- **Maths:** `+ - * /` work element by element, with the same broadcasting rules as [[NumPy]]. `@` is the matrix product. Reductions such as `.sum()`, `.mean()` and `.argmax()` take a `dim` (NumPy calls it `axis`).
- **Sharing with NumPy:** `torch.from_numpy(arr)` makes a tensor that **shares memory** with the array. Changing one changes the other (the code shows this). Use `.clone()` for a separate copy.
- **GPU:** `x.to('cuda')` moves a tensor to a GPU, if you have one. Both operands of an operation must be on the same device. All code in this note ran on a CPU (see [[GPU]]).
- **Autograd:** a tensor made with `requires_grad=True` is watched. After `loss.backward()`, each watched tensor has a `.grad` holding its slope. Only tensors that are "leaves" (made by you, not computed) keep `.grad` by default.
- **Gradients accumulate:** each `backward()` *adds* to `.grad`. You must reset it (`optimizer.zero_grad()`) before each step, or the slopes from earlier steps pile up (shown in the code and in equation 3 below). Forgetting this is one of the most common bugs.
- **`torch.no_grad()`:** a block in which nothing is recorded. Use it when running a trained model (it saves memory and time). `.detach()` cuts one tensor out of the graph, and `.item()` turns a one-number tensor into a plain Python number.
- **`nn.Module`:** the base class for models. A model holds *parameters* (the learnable tensors) and a `forward` method. Ready-made layers such as `nn.Linear`, `nn.ReLU` and `nn.Dropout` can be combined with `nn.Sequential`. `model.parameters()` lists everything the optimiser should update.
- **Train and eval modes:** some layers behave differently while training and when testing. [[Dropout]] switches parts off at random only in training mode (`model.train()`), and is off in `model.eval()`. Forgetting `eval()` gives noisy test results (the code shows this).
- **The standard loop** (five lines, in this order): `optimizer.zero_grad()`, `loss = loss_fn(model(x), y)`, `loss.backward()`, `optimizer.step()`, repeat. Each line does one job from the "How it works" list.
- **Why it is popular:** the graph is built *as the code runs* ("eager" mode), so ordinary Python (`if`, `for`, `print`) works inside a model, and debugging is easy (from memory; a design choice that other libraries have also adopted).
- **Reproducibility:** `torch.manual_seed(0)` fixes the random starting weights, so a run can be repeated. Exact numbers can still differ between versions and machines.
- **Versions:** the numbers below were run with PyTorch 2.8.0 (CPU), NumPy 1.24.4 and Python 3.9.

The code runs six small checks: (1) tensors and sharing memory with NumPy, (2) autograd against hand-worked slopes, (3) gradients piling up, (4) one gradient step by hand against the optimiser, (5) the standard training loop on noisy data, and (6) parameters, train and eval mode, and `no_grad`.

```python
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0)
# 1) tensors: shape, dtype, maths, broadcasting, sharing memory with NumPy
a = torch.tensor([[1., 2.], [3., 4.]]); b = torch.ones(2, 2) * 10
print(a.shape, a.dtype, '|', (a @ a).tolist(), '|', (a + torch.tensor([100., 200.])).tolist())      # @ is matrix product; the row is stretched over both rows
n = np.zeros(3); t = torch.from_numpy(n); t += 1; print('numpy array changed too:', n.tolist(), '| int64 default for whole numbers:', torch.tensor([1, 2]).dtype)
# 2) autograd: ask for derivatives, the graph is recorded as you compute
x = torch.tensor(3.0, requires_grad=True); y = x ** 2 + 2 * x; y.backward(); print('d/dx (x^2+2x) at 3 =', x.grad.item())
w1 = torch.tensor(0.5, requires_grad=True); w2 = torch.tensor(2.0, requires_grad=True); xin, tgt = 3.0, 2.0
h = torch.relu(w1 * xin); yhat = w2 * h; loss = (yhat - tgt) ** 2; loss.backward()
print('loss', loss.item(), '| dL/dw2', w2.grad.item(), '| dL/dw1', w1.grad.item(), '(by hand: 3.0 and 12.0)')
# 3) gradients add up unless you reset them
w = torch.tensor(1.0, requires_grad=True)
for k in (1, 2, 3): (w * 5).backward(); print('after', k, 'backward calls, w.grad =', w.grad.item())
w.grad.zero_(); (w * 5).backward(); print('after zero_ and one call:', w.grad.item())
# 4) one gradient step by hand: L = mean((w*x - y)^2), data y = 2x
xs = torch.tensor([1., 2., 3.]); ys = torch.tensor([2., 4., 6.]); w = torch.tensor(0.0, requires_grad=True)
L = ((w * xs - ys) ** 2).mean(); L.backward(); g = w.grad.item(); lr = 0.05
print('loss', round(L.item(), 3), '| gradient', round(g, 3), '| new w', round(0.0 - lr * g, 4))
opt = torch.optim.SGD([w], lr=lr); opt.step(); print('optimizer gives the same w:', round(w.item(), 4))
# 5) the standard training loop: fit y = 3x + 1 with noise
X = torch.linspace(-1, 1, 100).unsqueeze(1); Y = 3 * X + 1 + 0.1 * torch.randn(100, 1)
model = nn.Linear(1, 1); opt = torch.optim.SGD(model.parameters(), lr=0.1); lossf = nn.MSELoss()
for step in range(200):
    opt.zero_grad(); loss = lossf(model(X), Y); loss.backward(); opt.step()
print('learned weight and bias:', round(model.weight.item(), 2), round(model.bias.item(), 2), '| final loss', round(loss.item(), 4))
# 6) a small network: count the parameters, and train/eval mode matters for dropout
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 3))
print('parameters:', sum(p.numel() for p in net.parameters()), '(= 4*8+8 + 8*3+3)')
inp = torch.ones(1, 4); net.train(); o1, o2 = net(inp), net(inp); net.eval(); e1, e2 = net(inp), net(inp)
print('train mode: two calls differ:', not torch.allclose(o1, o2), '| eval mode: two calls equal:', torch.allclose(e1, e2))
with torch.no_grad(): z = net(inp)
print('inside no_grad, requires_grad:', z.requires_grad, '| normally:', net(inp).requires_grad)
# torch.Size([2, 2]) torch.float32 | [[7.0, 10.0], [15.0, 22.0]] | [[101.0, 202.0], [103.0, 204.0]]
# numpy array changed too: [1.0, 1.0, 1.0] | int64 default for whole numbers: torch.int64
# d/dx (x^2+2x) at 3 = 8.0
# loss 1.0 | dL/dw2 3.0 | dL/dw1 12.0 (by hand: 3.0 and 12.0)
# after 1 backward calls, w.grad = 5.0
# after 2 backward calls, w.grad = 10.0
# after 3 backward calls, w.grad = 15.0
# after zero_ and one call: 5.0
# loss 18.667 | gradient -18.667 | new w 0.9333
# optimizer gives the same w: 0.9333
# learned weight and bias: 3.0 1.0 | final loss 0.0105
# parameters: 67 (= 4*8+8 + 8*3+3)
# train mode: two calls differ: True | eval mode: two calls equal: True
# inside no_grad, requires_grad: False | normally: True
```
Reading the output:
- **Tensors:** the matrix product of `[[1,2],[3,4]]` with itself is `[[7,10],[15,22]]`. Adding the row `[100, 200]` stretches it over both rows. A NumPy array turned into a tensor with `from_numpy` is changed when the tensor is changed.
- **Autograd:** for $y=x^2+2x$, the slope at $x=3$ is $2x+2=8$. For the small network, autograd gives $3.0$ and $12.0$, the same as the by-hand chain rule below (shown in equation 2).
- **Piling up:** the slope 5 becomes 10 and then 15 after repeated `backward()` calls with no reset, and goes back to 5 after `zero_()`.
- **One step:** the loss is 18.667, the slope is −18.667, and one step of size 0.05 moves $w$ from 0 to 0.9333. The optimiser gives the same number. (That the loss and the slope have the same size here is a coincidence of this data, where $y=2x$ and $w=0$.)
- **Training loop:** 200 steps recover the true line $y=3x+1$ from noisy data: weight 3.0 and bias 1.0, after rounding, with a final loss of 0.0105, close to the noise level (the noise has spread 0.1, so its squared size is about 0.01).
- **Layers:** the small network has 67 learnable numbers. In train mode two calls give different results (dropout), and in eval mode they are the same. Inside `no_grad`, no graph is recorded.
- **Limits of these checks:** they are tiny problems on a CPU. They show how the pieces behave and not how fast PyTorch is, how it uses a GPU, or how it behaves on large models.

## 🔴 4. Deep / formal

**One step of learning.** To make the loss smaller, move each learnable number $w$ a small step against its slope:

$$w\leftarrow w-\eta\,\frac{\partial L}{\partial w},\qquad L(w)=\frac1n\sum_{i=1}^{n}(w x_i-y_i)^2\ \Rightarrow\ \frac{\partial L}{\partial w}=\frac2n\sum_{i=1}^{n}x_i\,(w x_i-y_i)$$

**Where:**
- $w$ is the learnable number (a weight). **Matters:** it is the knob being tuned.
- $\eta$ ("eta") is the learning rate, a small positive number. **Matters:** too small and learning crawls, too large and the steps overshoot the best value (see [[Learning Rate]]).
- $\partial L/\partial w$ is the slope of the loss with respect to $w$: how much $L$ changes per unit change of $w$. **Matters:** its sign gives the direction to move (a negative slope means "increase $w$"), and its size says how strongly.
- $L(w)$ is the mean squared error over $n$ examples $(x_i,y_i)$. **Matters:** it turns "how wrong is the model overall?" into one number (see [[Loss Function]] and [[Linear Regression]]).
- The minus sign. **Matters:** it makes the step go downhill. Without it, the loss would grow.
- PyTorch computes $\partial L/\partial w$ for you with `backward()`, and `optimizer.step()` does the update.

**Example (tiny numbers):** $x=(1,2,3)$, $y=(2,4,6)$ (the true rule is $y=2x$), start at $w=0$, $\eta=0.05$.
1. Loss: $\frac13(4+16+36)=18.667$.
2. Slope: $\frac23\big(1\cdot(0-2)+2\cdot(0-4)+3\cdot(0-6)\big)=\frac23(-28)=-18.667$.
3. Step: $w\leftarrow0-0.05\times(-18.667)=0.9333$.
4. The code gives the same slope and the same new $w$ (0.9333), with and without the optimiser.
5. Meaning: the slope is negative, so raising $w$ lowers the loss. One step takes $w$ about halfway to the true value 2. Repeating the step walks $w$ toward 2.

**The chain rule through the graph.** When a result is built in steps, its slope with respect to an early number is the product of the slopes of each step. For the small network in the code ($h=\mathrm{relu}(w_1x)$, $\hat y=w_2h$, $L=(\hat y-t)^2$):

$$\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial h}\cdot\frac{\partial h}{\partial w_1}=2(\hat y-t)\cdot w_2\cdot x\quad(\text{when } w_1x>0)$$

**Where:**
- $\partial L/\partial\hat y=2(\hat y-t)$ is how fast the loss changes with the model's output. **Matters:** it is the starting slope, and the bigger the error, the bigger it is.
- $\partial\hat y/\partial h=w_2$ is how the output changes with the hidden value. **Matters:** it passes the slope backward through the second layer.
- $\partial h/\partial w_1=x$ when the ReLU is active (the input $w_1x$ is positive), and $0$ when it is switched off. **Matters:** a ReLU that outputs 0 blocks the slope completely, so $w_1$ gets no gradient. See [[Activation Function]].
- The product of the three. **Matters:** this is what `backward()` does automatically for every watched number, node by node, along the recorded graph. See [[Backpropagation]].

**Example (tiny numbers):** $w_1=0.5$, $w_2=2$, input $x=3$, target $t=2$.
1. Forward: $w_1x=1.5$ (positive, so ReLU keeps it), $h=1.5$, $\hat y=2\times1.5=3$, $L=(3-2)^2=1$.
2. $\partial L/\partial\hat y=2(3-2)=2$. $\partial\hat y/\partial h=w_2=2$. $\partial h/\partial w_1=x=3$.
3. Product: $2\times2\times3=12$. For $w_2$: $\partial L/\partial w_2=2\times h=2\times1.5=3$.
4. The code's autograd gives $12.0$ and $3.0$.
5. Meaning: raising $w_1$ by a tiny amount raises the loss 12 times as fast, and raising $w_2$ raises it 3 times as fast. A step against these slopes lowers the loss.

**Why gradients must be reset.** Each call to `backward()` adds to what is already stored in `.grad`:

$$g_{\text{stored}}\leftarrow g_{\text{stored}}+\frac{\partial L}{\partial w}$$

**Where:**
- $g_{\text{stored}}$ is the number kept in `.grad`. **Matters:** the optimiser reads it to decide the step, so a stale value changes the step.
- $\partial L/\partial w$ is the slope found by the current `backward()`. **Matters:** it is the correct new information.
- The `+` (adding, not replacing). **Matters:** it is on purpose, since it lets you add up the slopes from several small batches before taking one step. But if you do not reset with `zero_grad()` on each step of an ordinary loop, old slopes are mixed in and the steps go the wrong way or too far.

**Example (tiny numbers):** $L=5w$, so the slope is always 5.
1. After 1 call: $0+5=5$. After 2 calls: $5+5=10$. After 3 calls: $10+5=15$.
2. After `zero_()` and one call: $0+5=5$.
3. The code prints exactly these numbers.
4. Meaning: without a reset, the third step would use a slope of 15 although the true slope is 5, so it would take a step three times too large. Reset the gradients once per step (unless you are deliberately adding up several batches).

**Other points (from memory, not checked here):**
- PyTorch was developed at Facebook AI Research and released in 2016, and was influenced by an earlier library called Torch. I did not check either detail for this note.
- The idea of recording operations as they run and replaying them backward is called *reverse-mode automatic differentiation*. It is cheap when there are many inputs and one output (like millions of weights and one loss), which is why it suits learning.
- Other libraries have similar ideas (TensorFlow, JAX), with different styles of building the graph. I did not compare them here.
- Real projects also need data loading (`Dataset` and `DataLoader`), saving and loading models, and mixed precision (see [[Quantization]]). None of that is covered here.
- Performance on a GPU depends on the hardware and on keeping data on the device. I ran everything on a CPU.

**What came before, and how this improved on it:**
- **Before:** derive the gradient of every layer by hand, then write it as code ([[Backpropagation]]).
- **Its limit:** every new layer needs new derivations, and one slip gives a silently wrong model.
- **What changed:** record the operations as they run and compute the gradient automatically (autograd). PyTorch was released by Facebook's AI research group in 2016 and built on the older Torch library (from memory, not checked).
- **How much better:** in the code below, for a 2-layer network, one `loss.backward()` call matched the hand-derived gradient with a largest difference of 0.0e+00, and the hand version needed 2 lines of derivation for just two layers (and more with every layer added).

**Check it yourself:**

```python
import torch
torch.manual_seed(0)
x = torch.randn(5, 3); y = torch.randn(5, 1)
W1 = torch.randn(3, 4, requires_grad=True); W2 = torch.randn(4, 1, requires_grad=True)
h = torch.tanh(x @ W1); loss = ((h @ W2 - y) ** 2).mean()
loss.backward()                                                  # autograd: one call
g = 2 * (h @ W2 - y) / 5                                         # the same gradients by hand: five lines
gW2 = h.T @ g; gh = g @ W2.T; gW1 = x.T @ (gh * (1 - h ** 2))
print("largest difference, autograd vs hand-derived gradient:", f"{max((W1.grad - gW1).abs().max().item(), (W2.grad - gW2).abs().max().item()):.1e}")
print("lines to write for the gradient: autograd 1 call, by hand 2 lines for this 2-layer net, and more for every layer you add")
```

## Related
[[Python]] · [[NumPy]] · [[Pandas]] · [[Matplotlib]] · [[Neural Network]] · [[Backpropagation]] · [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Adam Optimizer]] · [[Loss Function]] · [[Learning Rate]] · [[Activation Function]] · [[Dropout]] · [[Linear Regression]] · [[GPU]] · [[Matrix]] · [[TensorFlow]] · [[Git]] · [[Command Line]] · [[Virtual Environments]]
- [[FastAPI]] · [[Docker]]
- [[Snake DQN]]
## Sources
- PyTorch documentation: *Learn the Basics* and *Autograd mechanics* (pytorch.org).
- Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, 2019 (from memory; I did not open it for this note).
- Figure: [Wikimedia Commons, Computation_graph_example.svg](https://commons.wikimedia.org/wiki/File:Computation_graph_example.svg), by BrentPrinciple, CC BY-SA 4.0. I viewed it before writing the caption.
- The code and every Example number were run for this note (PyTorch 2.8.0 CPU, NumPy 1.24.4, Python 3.9). The data are tiny and made up for teaching.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dl-frameworks PyTorch:::A machine that remembers every knob you touched, measures how wrong the result is, and tells you which way to turn each knob.

#flashcards/programming/reverse #flashcards/topic/dl-frameworks Tensor:::A grid of numbers (a single number, a list, a table or a stack of tables) with a shape (size in each direction) and a dtype (decimals or whole numbers).

#flashcards/programming/concept #flashcards/topic/dl-frameworks What does `requires_grad` do?::It tells PyTorch to watch a tensor (such as a network weight) so it can find slopes for it.

#flashcards/programming/concept #flashcards/topic/dl-frameworks What is the computation graph?::As you compute, PyTorch quietly records each step in a chain, like breadcrumbs. `loss.backward()` walks this chain backwards.

#flashcards/programming/concept #flashcards/topic/dl-frameworks What does `loss.backward()` do?::It works out, for every watched number, how much the loss changes if that number changes a little. This is backpropagation done for you.

#flashcards/programming/concept #flashcards/topic/dl-frameworks List the steps of the training loop.::Run the model forward, measure the loss, `loss.backward()` for the slopes, take a small optimiser step against the slope (SGD or Adam), then repeat thousands of times.

#flashcards/programming/formula #flashcards/topic/dl-frameworks Write a gradient-descent step and the slope for a one-weight model and explain each symbol.
?
$$w\leftarrow w-\eta\,\frac{\partial L}{\partial w},\qquad L(w)=\frac1n\sum_{i=1}^{n}(w x_i-y_i)^2\ \Rightarrow\ \frac{\partial L}{\partial w}=\frac2n\sum_{i=1}^{n}x_i\,(w x_i-y_i)$$
- $w$: the learnable number (a weight), the knob being tuned.
- $\eta$: the learning rate, a small positive number. Too small and learning crawls, too large and the steps overshoot.
- $\partial L/\partial w$: the slope of the loss with respect to $w$. Its sign gives the direction to move (a negative slope means increase $w$).
- $x_i,y_i$: the inputs and targets. $n$: how many there are. The square makes big errors count more.

#flashcards/programming/example #flashcards/topic/dl-frameworks $x=(1,2,3)$, $y=(2,4,6)$, start $w=0$, $\eta=0.05$. What are the loss, slope and new $w$?::Loss $\frac13(4+16+36)=18.667$. Slope $\frac23(1\cdot(-2)+2\cdot(-4)+3\cdot(-6))=-18.667$. Step $w=0-0.05\times(-18.667)=0.9333$. The slope is negative, so raising $w$ lowers the loss. The note's code gave the same numbers with and without the optimiser.

#flashcards/programming/formula #flashcards/topic/dl-frameworks Write the chain rule for the slope of the first weight in a two-weight network and explain each symbol.
?
$$\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial h}\cdot\frac{\partial h}{\partial w_1}=2(\hat y-t)\cdot w_2\cdot x\quad(\text{when } w_1x>0)$$
- $\partial L/\partial\hat y=2(\hat y-t)$: how fast the loss changes with the model's output. It is the starting slope, bigger for bigger errors.
- $\partial\hat y/\partial h=w_2$: how the output changes with the hidden value. It passes the slope backward through the second layer.
- $\partial h/\partial w_1=x$ when the ReLU is active (the input $w_1x$ is positive) and 0 when it is switched off.
- $t$: the target. $h$: the hidden value.

#flashcards/programming/example #flashcards/topic/dl-frameworks $w_1=0.5$, $w_2=2$, $x=3$, $t=2$. What are the two slopes?::Forward: $w_1x=1.5$ (ReLU keeps it), $h=1.5$, $\hat y=3$, $L=1$. $\partial L/\partial\hat y=2$, $\partial\hat y/\partial h=2$, $\partial h/\partial w_1=3$. Product $2\times2\times3=12$. For $w_2$: $2\times1.5=3$. The note's autograd gave 12.0 and 3.0.

#flashcards/programming/formula #flashcards/topic/dl-frameworks Write what `backward()` does to the stored gradient and explain each symbol.
?
$$g_{\text{stored}}\leftarrow g_{\text{stored}}+\frac{\partial L}{\partial w}$$
- $g_{\text{stored}}$: the number kept in `.grad`. The optimiser reads it to decide the step, so a stale value changes the step.
- $\partial L/\partial w$: the slope found by the current `backward()`.
- The `+` (adding, not replacing) is on purpose: it lets you add slopes from several small batches before taking one step. But if you do not reset with `zero_grad()` on each step, old slopes pile up.

#flashcards/programming/example #flashcards/topic/dl-frameworks $L=5w$, so the slope is always 5. What is `.grad` after 1, 2 and 3 calls, and after a reset?::5, 10, 15. After a reset (`zero_()`) and one call: 5. Without a reset the third step would use 15 though the true slope is 5, a step three times too large.

#flashcards/programming/image #flashcards/topic/dl-frameworks ![[pytorch-computation-graph.svg]]
What does this picture show?
?
A computation graph in general: the arrows show which results feed into which. Here the two multiplications (x = 2×5 and y = 5×7) both feed the final addition x + y. PyTorch records such a graph as you compute, then walks it backwards to get the slopes. The diagram is a general graph, not drawn by PyTorch itself. (From the note's caption: BrentPrinciple, Wikimedia Commons.)

#flashcards/programming/cloze #flashcards/topic/dl-frameworks Reset the gradients with ==`zero_grad()`== once per step, unless you deliberately add up several batches.
