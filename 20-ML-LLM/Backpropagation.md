---
tags: [backpropagation, neural-network, deep-learning, optimization, calculus, math]
status: growing
level: 4
tier: 2
---

# Backpropagation

## 🟢 1. Simple (age 12)
Backpropagation is how a neural network **finds out who is to blame for a mistake**. The network gives a wrong answer, and then the blame is passed **backwards**, from the answer to every setting that helped make it. Each setting learns how much it should change to do better next time.

**Example:** A team of kids builds a Lego tower and it falls over. The teacher does not just say "bad tower". She walks back along the line: "the top piece was crooked, and that is because the middle piece was loose, and that is because the base was uneven." Each builder hears exactly how much they contributed, and fixes only their own part.

## 🔵 2. How it works
1. **Forward pass:** push an example through the [[Neural Network]] and get its answer.
2. **Measure the mistake** with a [[Loss Function]]: one number for how wrong the answer was.
3. **Go backwards one layer at a time.** At the last layer, ask: "if this layer's output went up a tiny bit, how much would the loss change?"
4. Use that to answer the same question for the layer before, and the one before that, reusing the work already done (this is the chain rule from calculus).
5. Each weight ends up with a number: "how much the loss would change if I changed you a little". This list of numbers is the **gradient**.
6. [[Gradient Descent]] then nudges each weight a small step in the direction that lowers the loss.
7. Repeat on millions of examples.

The key trick: going backwards through the whole network costs about the same as going forward once. Without that, training a model with billions of weights would be impossible.

No image: I found no on-topic, freely licensed diagram on Wikimedia Commons (searches for the term mostly return unrelated files). The [[Neural Network]] note has the layer picture to follow along with.

## 🟠 3. Technical detail
- **Computation graph:** every operation (multiply, add, ReLU, [[Softmax]]) is a node. Forward fills each node with a value. Backward fills each node with a gradient.
- **Store, then reuse:** the forward pass must remember the intermediate values (activations), because the backward pass needs them. That is a big part of the memory a model needs in training (see [[GPU]]).
- **Each node only needs its local derivative:** it multiplies the gradient arriving from above by its own simple derivative, and passes the result down. That is why libraries can do it for you automatically (**autograd**).
- **For a layer $\mathbf{y}=W\mathbf{x}$**, the gradient with respect to the weights is an outer product and the gradient with respect to the input is $W^\top$ times the incoming gradient. Both are [[Matrix]] multiplications, so they run well on [[GPU]]s.
- **Vanishing and exploding gradients:** multiplying many numbers smaller than 1 shrinks the signal toward 0, and many larger than 1 blows it up. Fixes include ReLU-like activations, careful starting weights, normalisation layers and residual connections (see [[Transformer]]).
- **Check your work:** compare the backprop gradient with a numerical estimate (change a weight by a tiny amount and see how the loss moves). This is called gradient checking.

```python
# tiny net: y = w2 * relu(w1 * x), loss = (y - t)^2
x, w1, w2, t = 1.0, 2.0, 3.0, 3.0
relu = lambda z: max(0.0, z)

# forward
h = relu(w1 * x); y = w2 * h; L = (y - t) ** 2
# backward (chain rule, one step at a time)
dy  = 2 * (y - t)               # dL/dy
dw2 = dy * h                    # dL/dw2
dh  = dy * w2                   # dL/dh
dw1 = dh * (1.0 if w1 * x > 0 else 0.0) * x   # dL/dw1 (ReLU slope is 1 when positive)
print(L, dw1, dw2)              # 9.0 18.0 12.0

# gradient check with a tiny nudge
f = lambda a, b: (b * relu(a * x) - t) ** 2
e = 1e-6
print(round((f(w1 + e, w2) - f(w1 - e, w2)) / (2 * e), 3))   # 18.0
```

## 🔴 4. Deep / formal
**The chain rule.** If the loss $L$ depends on a weight $w$ only through a chain of values, the effect of $w$ on $L$ is the product of the effects along the chain. For a single neuron $y=wx+b$:

$$
\frac{\partial L}{\partial w}=\frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial w},\qquad
\frac{\partial L}{\partial b}=\frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial b}
$$

**Where:**
- $L$ is the loss, $y$ is the neuron's output, $w$ is its weight, $x$ its input, and $b$ its bias.
- $\dfrac{\partial L}{\partial w}$ reads "how much does $L$ change if I change $w$ a tiny bit (everything else fixed)?". This is the number [[Gradient Descent]] needs. **Matters:** its sign says which way to move $w$, and its size says how strongly $w$ affects the mistake.
- $\dfrac{\partial L}{\partial y}$ is the blame arriving **from the later part** of the network. **Matters:** this is the quantity that is passed backwards. Everything upstream reuses it instead of recomputing from scratch.
- $\dfrac{\partial y}{\partial w}$ is the **local** effect: for $y=wx+b$ it equals $x$ (the bigger the input, the more this weight matters). Likewise $\dfrac{\partial y}{\partial b}=1$.
- The dot $\cdot$ is ordinary multiplication: total effect equals blame from above times local effect.

**Example (tiny numbers):** $x=2$, $w=1$, $b=0$, target $t=5$, loss $L=(y-t)^2$.
- Forward: $y=1\cdot2+0=2$. Loss $L=(2-5)^2=\mathbf{9}$.
- Backward: $\partial L/\partial y=2(y-t)=2(2-5)=\mathbf{-6}$.
- Local effects: $\partial y/\partial w=x=2$ and $\partial y/\partial b=1$.
- So $\partial L/\partial w=-6\cdot2=\mathbf{-12}$ and $\partial L/\partial b=-6\cdot1=\mathbf{-6}$.
- Both are negative: increasing $w$ or $b$ would lower the loss. Take a step with learning rate $0.1$: $w=1-0.1\cdot(-12)=\mathbf{2.2}$ and $b=0-0.1\cdot(-6)=\mathbf{0.6}$.
- New prediction: $2.2\cdot2+0.6=5.0$, which is exactly the target, so the loss is $0$. (A lucky landing from these friendly numbers; usually you need many small steps.)

**Through several layers.** For the two-step network $h=\operatorname{ReLU}(w_1x)$, $y=w_2h$, the chain simply gets longer:

$$
\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial h}\cdot\frac{\partial h}{\partial w_1}
$$

**Where:**
- $L$, $y$, $h$, $w_1$, $w_2$, $x$ are the loss, output, hidden value and the weights and input of the small network above.
- $\dfrac{\partial L}{\partial y}$ is the blame at the output, as before.
- $\dfrac{\partial y}{\partial h}=w_2$ is how much the output reacts to the hidden value. **Matters:** the blame is scaled by the weight connecting the layers, so a weight near 0 passes almost no blame back.
- $\dfrac{\partial h}{\partial w_1}$ is $x$ if $w_1x>0$ and $0$ otherwise (the ReLU slope). **Matters:** a ReLU that is switched off (negative input) blocks the blame completely, so weights feeding a "dead" neuron do not learn.

**Example (tiny numbers):** $x=1$, $w_1=2$, $w_2=3$, target $t=3$.
- Forward: $h=\max(0,2\cdot1)=2$. $y=3\cdot2=6$. $L=(6-3)^2=\mathbf{9}$.
- $\partial L/\partial y=2(6-3)=6$. $\partial y/\partial h=w_2=3$. $\partial h/\partial w_1=x=1$ (since $2>0$).
- $\partial L/\partial w_1=6\cdot3\cdot1=\mathbf{18}$. Also $\partial L/\partial w_2=6\cdot h=6\cdot2=\mathbf{12}$.
- Step with learning rate $0.01$: $w_1=2-0.18=1.82$, $w_2=3-0.12=2.88$. New output $2.88\cdot1.82=5.24$, loss $=\mathbf{5.02}$, down from $9$.
- Meaning: both weights are too big, so both get pushed down, and the loss drops after one step.

**Further notes:**
- **Reverse-mode automatic differentiation:** backprop is one case of this general technique, which gives the gradient of one output with respect to all inputs in about the cost of one forward pass. It was developed independently several times, and the paper by Rumelhart, Hinton and Williams (1986) popularised it for neural networks. (History stated from memory; check.)
- **Memory vs compute:** *gradient checkpointing* stores only some activations and recomputes the rest during the backward pass, trading extra compute for less memory.
- **Not biologically realistic:** the brain is not believed to pass exact error signals backwards like this. Whether it does something similar is an open question.

**What came before, and how this improved on it:**
- **Before:** to find how the loss changes with each weight, nudge each weight a little and re-run the network (numerical differentiation). One weight needs two forward passes.
- **Its limit:** the cost grows with the number of weights $P$: $2P$ forward passes for one gradient, which is hopeless for millions of weights.
- **What changed:** apply the chain rule backwards from the loss, reusing each intermediate result, so one forward pass and one backward pass give every gradient (Rumelhart, Hinton and Williams, 1986; the technique was found several times independently, as the note above says; dates from memory).
- **How much better:** the code below builds a network with 241 weights. Numerical differentiation needs 482 forward passes, backprop needs one forward and one backward pass, and the two gradients agree to $1.5\times10^{-8}$. For models with billions of weights the gap is billions of times.

**Check it yourself:**

```python
import torch
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.Tanh(), torch.nn.Linear(20, 1)).double()
x, t = torch.randn(8, 10).double(), torch.randn(8, 1).double()
loss = lambda: ((net(x) - t) ** 2).mean()
P = sum(p.numel() for p in net.parameters())
loss().backward()
g_back = torch.cat([p.grad.flatten() for p in net.parameters()])
g_num, eps = [], 1e-6                      # two forward passes per parameter
for p in net.parameters():
    for i in range(p.numel()):
        old = p.data.view(-1)[i].item()
        p.data.view(-1)[i] = old + eps; a = loss().item()
        p.data.view(-1)[i] = old - eps; b = loss().item()
        p.data.view(-1)[i] = old
        g_num.append((a - b) / (2 * eps))
print("parameters:", P)
print("forward passes, numerical:", 2 * P, " backprop: 1 forward + 1 backward")
print("max gradient difference:", float((torch.tensor(g_num) - g_back).abs().max()))
```

## Related
- [[Neural Network]] · [[Loss Function]] · [[Gradient Descent]] · [[Softmax]] · [[Matrix]] · [[GPU]] · [[Transformer]] · [[Vanishing and Exploding Gradients]] · [[Activation Function]] · [[PyTorch]] · [[Recursion]] · [[TensorFlow]] · [[Perceptron]] · [[Forward Propagation]] · [[ReLU]] · [[Gradient Checkpointing]] · [[Chain Rule]] · [[Jacobian]]
- [[Saliency Map]]
## Sources
- Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors*, 1986 (from memory)
- 3Blue1Brown, *Neural Networks*, chapters on backpropagation (video)
- Nielsen, *Neural Networks and Deep Learning*, chapter 2 (free online)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Backpropagation:::How a network finds out who is to blame for a mistake. The blame is passed backwards from the answer to every setting, so each setting learns how much to change.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the gradient of the loss for one neuron's weight and bias and explain each symbol.
?
$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial w},\qquad \frac{\partial L}{\partial b}=\frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial b}$$
- $L$: the loss. $y$: the neuron's output. $w$: its weight. $b$: its bias.
- $\partial L/\partial w$: how much $L$ changes if you change $w$ a tiny bit, everything else fixed. This is the number gradient descent needs. Its sign says which way to move $w$ and its size says how strongly $w$ matters.
- $\partial L/\partial y$: the blame at the output. $\partial y/\partial w$: the local effect of $w$ on the output.
- The pieces multiply, which is the chain rule.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the gradient for a weight in the first layer of a two-layer network and explain each symbol.
?
$$\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial h}\cdot\frac{\partial h}{\partial w_1}$$
- $L$: loss. $y$: output. $h$: hidden value. $w_1$: the first-layer weight.
- $\partial L/\partial y$: the blame at the output.
- $\partial y/\partial h=w_2$: how much the output reacts to the hidden value. The blame is scaled by the weight connecting the layers.
- $\partial h/\partial w_1$: how much the hidden value reacts to $w_1$. Each layer adds one more factor, going backwards.

#flashcards/ml/cloze #flashcards/topic/neural-nets Backpropagation passes the blame ==backwards== using the ==chain rule==, so each factor is a local effect.

#flashcards/ml/example #flashcards/topic/neural-nets $x=2$, $w=1$, $b=0$, target $t=5$, loss $L=(y-t)^2$. What are $\partial L/\partial w$ and $\partial L/\partial b$?::y = 2, L = 9. ∂L/∂y = 2(2 − 5) = −6. ∂y/∂w = x = 2, ∂y/∂b = 1. So ∂L/∂w = −12 and ∂L/∂b = −6. Both negative, so increase w and b.

#flashcards/ml/example #flashcards/topic/neural-nets $x=1$, $w_1=2$, $w_2=3$, target $3$, ReLU hidden. What is $\partial L/\partial w_1$?::h = 2, y = 6, L = 9. ∂L/∂y = 6, ∂y/∂h = 3, ∂h/∂w₁ = 1. So ∂L/∂w₁ = 6·3·1 = 18. (∂L/∂w₂ = 6·h = 12.)
