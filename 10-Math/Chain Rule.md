---
tags: [math, calculus, derivative, backpropagation, deep-learning, optimization]
status: growing
level: 4
tier: 2
---

# Chain Rule

The chain rule tells you **how fast the end of a chain of steps changes when you nudge the start**: multiply how fast each step changes. It is the one rule that makes [[Backpropagation]] work.

## 🟢 1. Simple (age 12)
When one thing depends on a second thing, and the second depends on a third, you can find how the first reacts to the third by **multiplying the two reactions together**.

**Example:** You earn 5 stickers for every book you read. A shop gives you 2 coins for every sticker. How many more coins do you get for reading one more book? The first step turns 1 book into 5 stickers. The second turns each sticker into 2 coins. So 1 more book gives $5\times2=10$ more coins. You multiplied the two rates. That is the chain rule.

## 🔵 2. How it works
1. Look at a calculation made of small steps. For example: take $x$, multiply by 3 and add 1 to get $y$, then square $y$ to get $z$. The steps form a chain: $x\to y\to z$.
2. For each step on its own, find its **rate of change** (its [[Calculus]] "derivative"): how much the step's output moves when its input is nudged a tiny bit.
3. **Multiply the rates** along the chain. The product is the rate of change of the whole chain.
4. When a step has many inputs and many outputs, the single rates become a table of rates. That table is the [[Jacobian]]. The idea is the same: multiply along the chain (with matrices).
5. A [[Neural Network]] is a long chain: input, layer 1, layer 2, ..., prediction, then the [[Loss Function]]. To learn, we need "how does the loss change if I nudge this one weight?" The chain rule answers that by multiplying the rates from the loss back to the weight.
6. Doing this from the end of the chain back to the start, and reusing the work already done, is [[Backpropagation]].
7. A warning: if many of the rates in the chain are small (say 0.25 each), their product shrinks toward zero. That is the [[Vanishing and Exploding Gradients]] problem.

![[chain-rule-animation.gif]]
*An animation of the chain rule along three axes: a curve takes $x$ to $y$ (the curve on the floor), and a second curve takes $y$ to $z$ (the curve in the vertical plane). The blue lines link the two steps. I viewed only one frame of the animation. Image: Brnbrnz, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Chain_rule_animation.gif), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Computation graph:** a program is a graph of small operations. Each operation knows its own local rate. The chain rule combines them.
- **Forward pass, then backward pass:** the forward pass computes and remembers every intermediate value. The backward pass multiplies the local rates from the loss back to each weight. This is "reverse mode" automatic differentiation, the kind that [[Backpropagation]] and PyTorch's autograd use.
- **Branches add up:** if a value feeds two later steps, the rates along both paths are added (the multi-variable form below).
- **Why not nudge each weight?** You can estimate a derivative by nudging a weight a tiny bit and measuring the loss again. But a network with a million weights would need a million extra forward passes per step. The chain rule gives all of them from one backward pass.
- **Vanishing gradients:** a sigmoid's rate is at most 0.25, so a chain of ten of them multiplies to 0.25 to the power 10, which is about $10^{-6}$. See [[Activation Function]] and [[ReLU]] for the fix, and [[Residual Connection]] for another.

```python
import numpy as np, torch

# one neuron:  a = w*x + b,  z = sigmoid(a),  loss = (z - y)^2
x, y, w, b = 2.0, 1.0, 0.5, -1.0
a = w * x + b
z = 1 / (1 + np.exp(-a))
loss = (z - y) ** 2
dL_dz = 2 * (z - y)          # outer link
dz_da = z * (1 - z)          # middle link
da_dw = x                    # inner link
print(a, z, loss, dL_dz, dz_da, da_dw, dL_dz * dz_da * da_dw)    # chain rule: multiply the links

h = 1e-6                     # check by nudging w a tiny bit
loss2 = (1 / (1 + np.exp(-((w + h) * x + b))) - y) ** 2
print(round((loss2 - loss) / h, 4))

t = torch.tensor(w, requires_grad=True)       # check with PyTorch autograd
((torch.sigmoid(t * x + b) - y) ** 2).backward()
print(t.grad.item())

print(0.25 ** 10)            # ten sigmoid links at their best (0.25 each)

# nudging every weight one by one vs one backward pass (float64, small network)
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double()
X, Y = torch.randn(5, 10).double(), torch.randn(5, 1).double()
f = lambda: ((net(X) - Y) ** 2).mean()
f().backward()
params = [p for p in net.parameters()]
P = sum(p.numel() for p in params)
base = f().item(); calls = 1; worst = 0.0
for p in params:
    for i in range(p.numel()):
        old = p.data.view(-1)[i].item()
        p.data.view(-1)[i] = old + 1e-6
        num = (f().item() - base) / 1e-6; calls += 1
        p.data.view(-1)[i] = old
        worst = max(worst, abs(num - p.grad.view(-1)[i].item()))
print(P, calls, worst < 1e-4)
```

Output: `0.0 0.5 0.25 -1.0 0.25 2.0 -0.5`, then `-0.5` (nudging), then `-0.5` (autograd), then `9.5367431640625e-07`, then `1441 1442 True`. The three ways of finding the slope agree on $-0.5$. The last line says the small network has 1,441 weights, that checking each by nudging took 1,442 forward passes, and that the nudged slopes matched the backward-pass slopes.

## 🔴 4. Deep / formal
**One variable.** If $z$ depends on $y$, and $y$ depends on $x$, then:

$$
\frac{dz}{dx}=\frac{dz}{dy}\cdot\frac{dy}{dx}
$$

**Where:**
- $x$ is the start of the chain, $y$ the middle value and $z$ the end value.
- $\dfrac{dy}{dx}$ is how much $y$ moves per tiny nudge of $x$. $\dfrac{dz}{dy}$ is how much $z$ moves per tiny nudge of $y$.
- $\dfrac{dz}{dx}$ is the total rate: how much $z$ moves per tiny nudge of $x$.
- **Matters:** the rates multiply because a nudge of $x$ is first scaled by $dy/dx$ on its way to $y$, and that scaled nudge is scaled again by $dz/dy$. If any link is 0 the whole product is 0, and if any link is huge or tiny the product is too (the source of exploding and vanishing gradients).

**Example (tiny numbers):** $y=3x+1$ and $z=y^2$, at $x=2$.
- Value: $y=3\times2+1=7$.
- Links: $dy/dx=3$ and $dz/dy=2y=14$.
- Product: $14\times3=\mathbf{42}$.
- Check: $z=(3x+1)^2$ gives $2(3x+1)\times3=2\times7\times3=42$. ✓
- Meaning: near $x=2$, nudging $x$ up by 0.01 raises $z$ by about 0.42.

**Many variables.** If the middle layer has several values $y_1,\dots,y_m$ and a final number $L$ (the loss), the rates along every path add up:

$$
\frac{\partial L}{\partial x_j}=\sum_{i=1}^{m}\frac{\partial L}{\partial y_i}\,\frac{\partial y_i}{\partial x_j}\qquad\Longleftrightarrow\qquad \nabla_x L=J^{\top}\,\nabla_y L
$$

**Where:**
- $x_j$ is the $j$-th input and $y_i$ the $i$-th middle value. $L$ is the final number we care about.
- $\dfrac{\partial L}{\partial y_i}$ is how much $L$ cares about $y_i$. $\dfrac{\partial y_i}{\partial x_j}$ is how much $y_i$ reacts to $x_j$.
- $J$ is the [[Jacobian]]: the table whose entry in row $i$, column $j$ is $\partial y_i/\partial x_j$. $J^{\top}$ is that table flipped.
- $\nabla_y L$ is the list of all $\partial L/\partial y_i$ (the gradient arriving from later in the chain). $\nabla_x L$ is the gradient we want.
- **Matters:** the sum is there because $x_j$ can reach $L$ through several middle values. Skip a path and the gradient is wrong. In a network, this one line is repeated layer by layer, flipping the Jacobian each time. It is [[Backpropagation]].

**Example (tiny numbers):** $y_1=x_1x_2$, $y_2=x_1+x_2^2$ and $L=y_1+2y_2$, at $(x_1,x_2)=(2,3)$.
- $\nabla_yL=(1,\,2)$.
- $J=\begin{pmatrix}x_2&x_1\\1&2x_2\end{pmatrix}=\begin{pmatrix}3&2\\1&6\end{pmatrix}$, so $J^{\top}=\begin{pmatrix}3&1\\2&6\end{pmatrix}$.
- $J^{\top}\nabla_yL=(3\times1+1\times2,\;2\times1+6\times2)=(\mathbf{5},\,\mathbf{14})$.
- Check: $L=x_1x_2+2x_1+2x_2^2$, so $\partial L/\partial x_1=x_2+2=5$ and $\partial L/\partial x_2=x_1+4x_2=14$. ✓
- Meaning: nudging $x_2$ moves $L$ nearly three times as much as nudging $x_1$.

**Further notes:**
- **Cost:** reverse mode gets the gradient of one output with respect to all inputs in a time of the same order as one forward pass (a standard result; from memory). That is why training a network with millions of weights is possible.
- **Memory:** the backward pass needs the forward values, so they are stored. [[Gradient Checkpointing]] trades extra compute for less stored memory.
- **Second derivatives** (the [[Hessian]]) use the chain rule again, with an extra term.

**What came before, and how this improved on it:**
- **Before:** estimating each derivative by nudging one weight at a time (finite differences).
- **Its limit:** a network with $P$ weights needs $P+1$ forward passes for one gradient, and the answer is only approximate.
- **What changed:** apply the chain rule from the loss backwards and reuse shared work, which is backpropagation. It was popularised for neural networks by Rumelhart, Hinton and Williams in 1986, and similar ideas appeared earlier (Linnainmaa 1970 and Werbos 1974; all from memory, not checked).
- **How much better:** in the code above, a network with 1,441 weights needed 1,442 forward passes to get its gradient by nudging, against one forward pass and one backward pass with the chain rule, and the two gradients matched (the check printed `True`).

## Related
- [[Calculus]] · [[Jacobian]] · [[Backpropagation]] · [[Gradient Descent]] · [[Neural Network]] · [[Loss Function]] · [[Forward Propagation]] · [[Activation Function]] · [[Vanishing and Exploding Gradients]] · [[Hessian]] · [[Gradient Checkpointing]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, section 6.5 (free online; from memory)
- Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors", 1986 (from memory)
- Image: Brnbrnz, Wikimedia Commons, CC BY-SA 4.0 (see the caption in section 2)
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/calculus-optimization Chain rule:::To find how the end of a chain of steps reacts to the start, multiply the rate of change of each step.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the chain rule for one variable and explain each symbol.
?
$$\frac{dz}{dx}=\frac{dz}{dy}\cdot\frac{dy}{dx}$$
- $x$: the start of the chain. $y$: the middle value. $z$: the end value.
- $dy/dx$: how much $y$ moves per tiny nudge of $x$. $dz/dy$: how much $z$ moves per tiny nudge of $y$.
- $dz/dx$: the total rate. The rates multiply because a nudge is scaled once on the way to $y$ and again on the way to $z$.
- If any link is 0 the whole product is 0. Huge or tiny links make a huge or tiny product (exploding or vanishing gradients).

#flashcards/math/example #flashcards/topic/calculus-optimization $y=3x+1$, $z=y^2$, $x=2$: what is $dz/dx$ by the chain rule?::y = 7, dy/dx = 3, dz/dy = 2y = 14. Product: 14 × 3 = 42. Check: z = (3x+1)^2 gives 2(3x+1)(3) = 42. Nudging x up by 0.01 raises z by about 0.42.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the chain rule for many variables, in sum form and matrix form, and explain each symbol.
?
$$\frac{\partial L}{\partial x_j}=\sum_{i=1}^{m}\frac{\partial L}{\partial y_i}\,\frac{\partial y_i}{\partial x_j}\qquad\Longleftrightarrow\qquad \nabla_x L=J^{\top}\,\nabla_y L$$
- $x_j$: the $j$-th input. $y_i$: the $i$-th middle value. $L$: the final number (the loss).
- $\partial L/\partial y_i$: how much $L$ cares about $y_i$. $\partial y_i/\partial x_j$: how much $y_i$ reacts to $x_j$.
- $J$: the Jacobian, entry $(i,j)$ is $\partial y_i/\partial x_j$. $J^{\top}$ is it flipped.
- $\nabla_y L$: the gradient arriving from later. $\nabla_x L$: the gradient we want.
- The sum is there because $x_j$ can reach $L$ through several middle values. This line, repeated layer by layer, is backpropagation.

#flashcards/math/example #flashcards/topic/calculus-optimization $y_1=x_1x_2$, $y_2=x_1+x_2^2$, $L=y_1+2y_2$ at $(2,3)$: what is $\nabla_xL$?::The gradient arriving is (1, 2). `J = [[3, 2], [1, 6]]`. J-transpose times (1, 2) = (3·1 + 1·2, 2·1 + 6·2) = (5, 14). Check: dL/dx1 = x2 + 2 = 5 and dL/dx2 = x1 + 4·x2 = 14. My arithmetic.

#flashcards/math/cloze #flashcards/topic/calculus-optimization Backpropagation is the ==chain rule== applied from the ==loss== back to each weight, reusing shared work.

#flashcards/math/concept #flashcards/topic/calculus-optimization Why use the chain rule instead of nudging each weight to get a gradient?
?
Nudging needs one extra forward pass per weight (P + 1 passes for P weights) and is only approximate. The chain rule gives every weight's gradient from one forward and one backward pass. In the note's test, 1,441 weights needed 1,442 forward passes by nudging, and the gradients matched.

#flashcards/math/example #flashcards/topic/calculus-optimization What is 0.25 to the power 10, and why does it matter?::About 9.5 × 10^-7 (my run). A sigmoid's rate is at most 0.25, so ten of them in a chain shrink the gradient to nearly nothing: the vanishing gradient problem.

#flashcards/math/image #flashcards/topic/calculus-optimization ![[chain-rule-animation.gif]]
What does this animation show?
?
The chain rule along three axes: one curve takes x to y, a second curve takes y to z, and the rates of the two steps multiply to give how z reacts to x.

#flashcards/math/concept #flashcards/topic/calculus-optimization In a computation graph, what happens when one value feeds two later steps?
?
The rates along both paths are added. This is why the many-variable chain rule has a sum. Missing a path gives a wrong gradient.
