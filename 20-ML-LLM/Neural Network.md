---
tags: [neural-network, deep-learning, machine-learning, LLM, math]
status: growing
level: 4
tier: 2
---

# Neural Network

## 🟢 1. Simple (age 12)
A neural network is a **machine made of many tiny calculators wired in layers**. Numbers go in one end, each calculator does a small bit of maths and passes its answer on, and a final answer comes out the other end. Nobody writes the rules. The network **tunes its own settings** until its answers are good.

**Example:** Imagine a team of judges deciding if a photo shows a cat. The first judges only look at tiny things ("is there a pointy shape?", "is there fur?"). The next judges listen to the first ones and decide bigger things ("that looks like an ear"). The last judge listens to everyone and says "cat". Training is the team practising on thousands of photos and slowly learning how much to trust each other.

## 🔵 2. How it works
1. **Input layer:** the data goes in as a list of numbers (a [[Vector]]). A photo becomes its pixel brightnesses; a word becomes an [[Embedding]].
2. **Each neuron** takes all the numbers coming in, multiplies each by a **weight** (how much it cares about that input), adds them up, and adds a **bias** (a nudge up or down).
3. **Activation:** the neuron then squashes or bends that total with a simple rule (for example "if negative, make it 0"). Without this bend, stacking layers would be pointless.
4. The neuron's answer goes to every neuron in the next layer. Layers in the middle are called **hidden layers**.
5. The **output layer** gives the final answer (a score, a yes/no, the next word).
6. **Training:** compare the answer with the right one using a [[Loss Function]], then use [[Backpropagation]] and [[Gradient Descent]] to nudge every weight a little so the next answer is better. Repeat millions of times.

![[neural-network-layers.svg]]
*A small network: 3 inputs (black dots), a hidden layer of 3 neurons, and an output layer of 2 neurons. Every arrow carries a number multiplied by a weight. Image: Offnfopt, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Multi-Layer_Neural_Network-Vector-Blank.svg), CC BY-SA 3.0.*

Why it matters: the [[Transformer]] behind every modern [[LLM]] is a neural network, with a clever [[Attention]] step added between layers.

## 🟠 3. Technical detail
- **Parameters:** all the weights and biases. A tiny network has dozens; a modern [[LLM]] has billions. Training means finding good values for them.
- **Layers as matrices:** a whole layer is one [[Matrix]] multiplication plus a bias, then the activation. That is why [[GPU]]s, which are very fast at matrix maths, are used.
- **Common activations:**
  - **ReLU**, $\max(0,z)$: cheap and the usual default.
  - **Sigmoid**, which squashes to 0 to 1: used when you want a probability for yes/no.
  - **GELU**: a smooth ReLU-like curve, common in transformers.
  - **[[Softmax]]** on the last layer turns several scores into probabilities that add to 1.
- **Forward pass:** push the input through the layers to get the output. **Backward pass:** push the error back through the layers to learn how to change each weight ([[Backpropagation]]).
- **Depth and width:** more layers (deeper) or more neurons per layer (wider) means a more flexible model, but more cost and more risk of overfitting (memorising the training data).
- **Kinds:** MLP (plain layers, as here), CNN (for images), RNN (for sequences, older), Transformer (current standard for text).

```python
import numpy as np
relu = lambda z: np.maximum(0, z)

x  = np.array([2., 1.])                       # input
W1 = np.array([[1., -1.], [0.5, 0.5]])        # hidden layer: 2 neurons
b1 = np.array([0., -1.])
W2 = np.array([[1., 2.]])                     # output layer: 1 neuron
b2 = np.array([0.5])

h = relu(W1 @ x + b1)
print(h)                                      # [1.  0.5]
print(W2 @ h + b2)                            # [2.5]
```

## 🔴 4. Deep / formal
**One neuron.** A neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function:

$$
y=\phi\big(\mathbf{w}\cdot\mathbf{x}+b\big)=\phi\Big(\sum_{i} w_i x_i+b\Big)
$$

**Where:**
- $\mathbf{x}=(x_1,x_2,\dots)$ is the list of input numbers coming into the neuron, and $y$ is its single output number.
- $\mathbf{w}=(w_1,w_2,\dots)$ are the **weights**, one per input. **Matters:** a big positive weight means "this input pushes me up strongly", a negative weight means "this input pushes me down", and near 0 means "ignore it". These are what training changes.
- $\mathbf{w}\cdot\mathbf{x}=\sum_i w_i x_i$ is the dot product (see [[Vector]]): multiply each input by its weight, then add everything up.
- $b$ is the **bias**, a constant added after the sum. **Matters:** it shifts the point at which the neuron "switches on". Without it the neuron's total would always be 0 when all inputs are 0.
- $\phi$ (phi) is the **activation function**, such as ReLU $\phi(z)=\max(0,z)$ or sigmoid $\phi(z)=1/(1+e^{-z})$. **Matters:** it is the bend. Without it, any stack of layers collapses into one big straight-line (linear) function and could not learn curves.

**Example (tiny numbers):** inputs $\mathbf{x}=(2,\,1)$, weights $\mathbf{w}=(0.5,\,-1)$, bias $b=0.5$.
- Weighted sum: $0.5\cdot2+(-1)\cdot1=1-1=0$.
- Add the bias: $z=0+0.5=\mathbf{0.5}$.
- With ReLU: $y=\max(0,0.5)=\mathbf{0.5}$. With sigmoid: $y=1/(1+e^{-0.5})=\mathbf{0.62}$.
- If the second input were $3$ instead of $1$: $z=1-3+0.5=-1.5$, so ReLU gives $\mathbf{0}$. The neuron switches off because input 2 (with its negative weight) wins.

**A whole layer, and a network.** Stack all neurons of a layer so their weights form a matrix. A two-layer network is then:

$$
\mathbf{h}=\phi\big(W_1\mathbf{x}+\mathbf{b}_1\big),\qquad \hat{\mathbf{y}}=W_2\,\mathbf{h}+\mathbf{b}_2
$$

**Where:**
- $\mathbf{x}$ is the input vector, $\mathbf{h}$ is the hidden layer's output vector, and $\hat{\mathbf{y}}$ ("y-hat") is the network's final answer (the hat means "predicted", not the true value).
- $W_1$ is the first layer's weight [[Matrix]]: one **row per neuron**, one **column per input**. $W_2$ is the same for the second layer. **Matters:** the matrix shape sets the layer sizes, and the matrix numbers are the learned knowledge.
- $\mathbf{b}_1,\mathbf{b}_2$ are the bias vectors, one number per neuron.
- $\phi$ is applied to each number of $W_1\mathbf{x}+\mathbf{b}_1$ separately. **Matters:** it sits between the two matrix steps. If you removed it, $W_2(W_1\mathbf{x})=(W_2W_1)\mathbf{x}$ would be just one matrix, so the hidden layer would add nothing.

**Example (tiny numbers):** $\mathbf{x}=(2,1)$, $W_1=\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix}$, $\mathbf{b}_1=(0,-1)$, $W_2=\begin{bmatrix}1&2\end{bmatrix}$, $\mathbf{b}_2=(0.5)$, $\phi=$ ReLU.
- Neuron 1: $1\cdot2+(-1)\cdot1+0=1$. Neuron 2: $0.5\cdot2+0.5\cdot1-1=0.5$. So $W_1\mathbf{x}+\mathbf{b}_1=(1,\,0.5)$.
- ReLU keeps both (they are positive): $\mathbf{h}=(1,\,0.5)$.
- Output: $1\cdot1+2\cdot0.5+0.5=\mathbf{2.5}$. So $\hat y=2.5$.
- If the right answer were $3$, the network is off by $0.5$. Training would nudge the weights to shrink that gap ([[Loss Function]], [[Gradient Descent]]).

**Further notes:**
- **Universal approximation:** a network with one wide enough hidden layer and a non-linear activation can approximate any reasonable function (Cybenko 1989, Hornik 1991; from memory, check). It says nothing about how easy it is to *find* the weights or how wide "wide enough" is.
- **Why depth helps:** deep networks reuse simple features to build complex ones, often needing far fewer neurons than a very wide shallow network for the same job.
- **Initialisation and vanishing gradients:** starting weights that are too big or too small make signals explode or fade as they pass through many layers, so training stalls. Careful initialisation, normalisation layers and residual connections (see [[Transformer]]) are the standard fixes.
- **Generalisation:** networks with more parameters than training examples can still generalise. This is only partly understood and is an active research area.

**What came before, and how this improved on it:**
- **Before:** a single artificial neuron (a perceptron, late 1950s; date from memory) that draws one straight line between two classes.
- **Its limit:** one line cannot separate "exclusive or" (XOR) data, a limit pointed out by Minsky and Papert in 1969 (from memory, not checked).
- **What changed:** stack neurons in layers with a non-linear activation, so the hidden layer builds new features that the next layer can cut with a line ([[Activation Function]]).
- **How much better:** in the code below, the best single line over a grid of 68,921 settings gets 0.75 on the four XOR points, and a hand-set network with 2 hidden neurons gets 1.0. A toy, but it is exactly the case the single neuron cannot do.

**Check it yourself:**

```python
import itertools, numpy as np
# XOR: best single neuron (a line) vs a 2-hidden-unit network
X = np.array([[0,0],[0,1],[1,0],[1,1]]); y = np.array([0,1,1,0])
best = 0
g = np.linspace(-2, 2, 41)
for w1, w2, b in itertools.product(g, g, g):
    best = max(best, ((X @ [w1, w2] + b > 0) == y).mean())
h = lambda x: np.array([x @ [1, 1] - 0.5 > 0, x @ [1, 1] - 1.5 > 0], float)  # OR, AND
out = lambda x: float(h(x) @ [1, -1] - 0.5 > 0)
print("best line accuracy:", best)
print("2-hidden-unit network accuracy:", np.mean([out(x) == t for x, t in zip(X, y)]))
```

## Related
- [[Vector]] · [[Matrix]] · [[Gradient Descent]] · [[Backpropagation]] · [[Loss Function]] · [[Softmax]] · [[Transformer]] · [[LLM]] · [[GPU]] · [[Embedding]] · [[Attention]] · [[Activation Function]] · [[Residual Connection]] · [[Convolutional Neural Network]] · [[Recurrent Neural Network]] · [[Generative Models]] · [[Linear Regression]] · [[Convex Optimization]] · [[Quantization]] · [[Autoencoder]] · [[GAN]] · [[PyTorch]] · [[TensorFlow]] · [[From FNN to RNN]] · [[History of AI]] · [[Perceptron]] · [[Forward Propagation]] · [[Machine Learning Specialization]] · [[ReLU]]
- [[Deep Q-Network]]
- [[Interpretability]] · [[Saliency Map]]
## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016 (free online)
- 3Blue1Brown, *Neural Networks* (video series)
- Nielsen, *Neural Networks and Deep Learning* (free online book)
- Cybenko, *Approximation by superpositions of a sigmoidal function*, 1989 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Neural network:::A machine made of many tiny calculators wired in layers. Numbers go in one end, each calculator passes a small result on, and an answer comes out. It tunes its own settings until its answers are good.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the output of one neuron and explain each symbol.
?
$$y=\phi\big(\mathbf{w}\cdot\mathbf{x}+b\big)=\phi\Big(\sum_{i} w_i x_i+b\Big)$$
- $\mathbf{x}=(x_1,x_2,\dots)$: the input numbers coming into the neuron. $y$: its single output number.
- $\mathbf{w}$: the weights, one per input. Big positive pushes the neuron up, negative pushes it down, near 0 means ignore that input. These are what training changes.
- $b$: the bias, a number added to the sum so the neuron can fire even when all inputs are 0.
- $\phi$: the activation function (for example ReLU or sigmoid) applied to the sum.

#flashcards/ml/formula #flashcards/topic/neural-nets Write a network with one hidden layer and explain each symbol.
?
$$\mathbf{h}=\phi\big(W_1\mathbf{x}+\mathbf{b}_1\big),\qquad \hat{\mathbf{y}}=W_2\,\mathbf{h}+\mathbf{b}_2$$
- $\mathbf{x}$: the input vector. $\mathbf{h}$: the hidden layer's output vector. $\hat{\mathbf{y}}$: the network's final answer (the hat means "predicted", not the true value).
- $W_1$, $W_2$: the weight matrices of the two layers, one row per neuron and one column per input.
- $\mathbf{b}_1$, $\mathbf{b}_2$: the bias vectors. $\phi$: the activation after the first layer.

#flashcards/ml/cloze #flashcards/topic/neural-nets ==Weights== are the numbers that training changes, and the ==bias== lets a neuron fire even when all inputs are 0.

#flashcards/ml/example #flashcards/topic/neural-nets A neuron has inputs (2, 1), weights (0.5, −1) and bias 0.5. What is the output with ReLU and with sigmoid?::Weighted sum 1 − 1 = 0, plus bias gives z = 0.5. ReLU: 0.5. Sigmoid: 1/(1 + e^−0.5) = 0.62.

#flashcards/ml/example #flashcards/topic/neural-nets For $\mathbf{x}=(2,1)$, $W_1=\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix}$, $\mathbf{b}_1=(0,-1)$, $W_2=\begin{bmatrix}1&2\end{bmatrix}$, $\mathbf{b}_2=0.5$ and ReLU, what is the output?::Neuron 1: 2 − 1 = 1. Neuron 2: 1 + 0.5 − 1 = 0.5. Both stay after ReLU. Output = 1·1 + 2·0.5 + 0.5 = 2.5.

#flashcards/ml/image #flashcards/topic/neural-nets ![[neural-network-layers.svg]]
What does this picture show?
?
A small network: 3 inputs, a hidden layer of 3 neurons and an output layer of 2 neurons. Every arrow carries a number multiplied by a weight.
