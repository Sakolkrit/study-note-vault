---
tags: [saliency, gradient-attribution, interpretability, backpropagation, neural-network, ML]
status: growing
level: 4
tier: 3
---

# Saliency Map

A saliency map is **a picture or a list that shows how strongly each input of a neural network pushes on its answer**. It is found by asking the network a calculus question: "if I nudge this input a tiny bit, how much does the output move?" Inputs that move the output a lot are called salient.

## 🟢 1. Simple (age 12)
Picture a sound mixing desk with many sliders, and a speaker. You want to know which slider controls the loudest part of the song. You **wiggle each slider a tiny bit and listen to how much the sound changes.** A slider that changes the sound a lot is important right now; a slider that changes nothing is not. A saliency map is the list of "how much did the sound change" for every slider.

**Example:** a game-playing network has the sliders "wall ahead", "food on the left" and so on, and the speaker is "how good is turning left". Wiggling "food on the left" changes the score a lot, so that input is salient at that moment.

## 🔵 2. How it works
1. **Take one input** to the network (an image, or a list of numbers) and the **one output** you care about (for example the score of the chosen move).
2. **Ask: how much does this output change if I nudge each input slightly?** Doing this by hand for every input would need one run per input.
3. **Backpropagation does it all at once.** The same trick that trains the network ([[Backpropagation]]) gives the answer for every input in a single backward pass.
4. **Take the size of each answer** (ignore whether it goes up or down) and show it: as a heat map over the pixels of an image, or as a bar per input number.
5. **Average over many inputs** if you want a general picture instead of one case.
6. **Check it.** A bright pixel or a tall bar says "the output is sensitive here", not necessarily "this is why the network decided" ([[Permutation Importance]] gives a second opinion).

*I found no free, on-topic picture of a gradient saliency map without a photograph of a real person in it (the only Commons example I saw was a diagram built around a celebrity photo), so this note has no picture.*

## 🟠 3. Technical detail
- **The gradient.** Let $Q(x)$ be the output of the network for the input vector $x$ (in [[Deep Q-Network]], the score of the best move). The saliency of input $j$ at this particular $x$ is the slope

$$
S_j(x)=\frac{\partial Q(x)}{\partial x_j}
$$

**Where:**
- $Q(x)$ is the single number being explained; $x_j$ is the $j$-th input.
- $\partial Q/\partial x_j$ is the partial derivative: the rate of change of $Q$ as only $x_j$ moves. It is the same quantity [[Gradient Descent]] uses for weights, but taken with respect to the *inputs*.
- **Matters:** it is **local**: it is the slope at this $x$ only. A big value means "a small change here changes the output a lot". A value of 0 can mean "does not matter" or "flat here, but matters elsewhere" (see the code below).

**Example (tiny numbers):** a made-up score $Q(x)=2x_1-x_2+0.5x_3$ at $x=(1,3,0)$.
- Slopes: $S=(2,\,-1,\,0.5)$, whatever the point (the score is a straight-line function).
- Raising $x_2$ by $0.1$ changes $Q$ by about $-1\times0.1=-0.1$.
- Meaning: $x_1$ is the strongest lever, and it pushes the score up; $x_2$ pushes it down.

- **Gradient times input.** The slope says how sensitive the output is; multiplying by the input's value says how much that input is contributing right now:

$$
A_j(x)=x_j\cdot\frac{\partial Q(x)}{\partial x_j}
$$

**Where:**
- $x_j$ is the input's current value and $\partial Q/\partial x_j$ its slope, as above.
- $A_j$ is the attribution: a signed amount of the score credited to input $j$.
- **Matters:** an input that is 0 gets no credit, however steep the slope (it is "off"). For a score with no constant term, such as the straight-line one above, the attributions add up to the score itself (my arithmetic below; I recall the same holds for some ReLU networks without bias terms, not checked).

**Example (tiny numbers):** the same $Q$ at $x=(1,3,0)$.
- $A=(1\times2,\;3\times(-1),\;0\times0.5)=(2,\,-3,\,0)$.
- Sum: $2-3+0=-1$, and $Q(x)=2\times1-3+0=-1$. They agree.
- Meaning: input 2 pulls the score down by 3, input 1 pushes it up by 2, and input 3, being 0, contributes nothing.

- **Code: the slope two ways, and a flat spot.** A small network (3 inputs, 4 tanh units, 1 output) is built with a fixed random seed. The slope is found by backpropagation, and again by nudging each input by $\pm0.001$ and dividing. The last lines show a function that is flat until $x=1$:

```python
# file: toy_saliency.py
import torch

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
x = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
net(x)[0].backward()                                                   # backpropagation gives dQ/dx for every input at once
grad = x.grad.clone(); eps = 1e-3; xd = x.detach()
with torch.no_grad():                                                  # nudge one input at a time instead
    fd = [float(net(xd + eps * e)[0] - net(xd - eps * e)[0]) / (2 * eps) for e in torch.eye(3)]
print("gradient by backpropagation:", [round(float(g), 4) for g in grad])
print("gradient by nudging +/-0.001:", [round(g, 4) for g in fd])
print("gradient x input:           ", [round(float(g * v), 4) for g, v in zip(grad, xd)])

relu = lambda t: torch.relu(t - 1.0)                                   # flat until t = 1, then rises
x0 = torch.tensor(0.0, requires_grad=True); relu(x0).backward()
print(f"flat region: at x = 0 the output is {float(relu(torch.tensor(0.0))):.0f} and the gradient is {float(x0.grad):.0f}, yet at x = 3 the output is {float(relu(torch.tensor(3.0))):.0f}")
```

```text
gradient by backpropagation: [-0.1581, -0.1087, 0.0711]
gradient by nudging +/-0.001: [-0.1581, -0.1087, 0.0711]
gradient x input:            [-0.079, 0.1087, 0.1422]
flat region: at x = 0 the output is 0 and the gradient is 0, yet at x = 3 the output is 2
```

- **Reading it:** both ways give the same slopes (−0.1581, −0.1087, 0.0711), so backpropagation really is a one-pass way to nudge every input. Gradient times input gives (−0.079, 0.1087, 0.1422). In the flat-region example, at $x=0$ the gradient is 0 and the output is 0, yet pushing $x$ to 3 gives 2: **a zero gradient does not prove the input is unimportant.**
- **What I measured on the Snake DQN.** See [[Looking Inside Snake DQN]]: the average size of the slope of the best move's score was largest for "danger ahead" (0.423) and smallest for the heading inputs (about 0.22 to 0.24), but the ranking of the 11 inputs agreed only weakly between five networks (mean Spearman correlation 0.46, lowest −0.01).

## 🔴 4. Deep / formal
- **Local, not global.** The slope is for this exact input. A different state can give a different picture, which is why I average its size over thousands of visited states, and even then the average describes sensitivity, not the network's overall logic.
- **Saturation and flat regions.** Where an activation is flat (a ReLU below its knee, a saturated tanh), the gradient is zero or tiny although the input could matter after a bigger change. The toy above shows it. Methods that integrate the slope along a path from a baseline input (integrated gradients, from memory, not run here) are one answer.
- **Sensitivity is not importance.** A large slope means the output *could* change a lot if the input changed a little. It does not say that the input in fact drives the decision (a one-hot input is 0 or 1, never "a little changed").
- **Noise and scale.** Saliency depends on the scale of the input. If an input is measured in metres instead of millimetres, its slope changes by 1,000 even though the network is the same.
- **Faithfulness versus plausibility.** A map can look sensible to a person and still not reflect what the network computes (a known worry, from memory). The checks that matter are tests such as removing or shuffling the inputs and watching the behaviour ([[Permutation Importance]]).
- **Images.** For pixels, the raw gradient map is often noisy; refinements such as SmoothGrad and Grad-CAM exist (from memory, not run here).
- **What I did not run:** integrated gradients, SmoothGrad, Grad-CAM, or any image network.

**What came before, and how this improved on it:**
- **Before:** treating a trained network as a black box and trying inputs one by one, or reading the first layer's weights.
- **Its limit:** one run per input is slow for thousands of pixels, and weights deep in a network say nothing about the input.
- **What changed:** use the gradient with respect to the input, which backpropagation provides for all inputs in a single pass. Gradient maps for image networks were popularised around 2013 to 2014 (from memory, not checked).
- **How much better:** measured in my toy: the one-pass gradients equal the nudge-by-nudge answers to four decimals (−0.1581, −0.1087, 0.0711) while needing 1 backward pass instead of 6 forward passes. The flat-region example shows the limit: a gradient of 0 where the output rises to 2 further on.

## Related
- [[Interpretability]] · [[Permutation Importance]] · [[Looking Inside Snake DQN]] · [[Backpropagation]] · [[Gradient Descent]] · [[Neural Network]] · [[Deep Q-Network]] · [[Convolutional Neural Network]] · [[ReLU]] · [[Chain Rule]] · [[Calculus]]

## Sources
- My own toy (PyTorch 2.8.0, Python 3.9.6) and the project [[Looking Inside Snake DQN]]. The 2013 to 2014 history, integrated gradients, SmoothGrad, Grad-CAM and the claim about attributions adding up for some ReLU networks are from memory; I did not read a source for this note.
- No image (see level 2).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Saliency map:::A list or picture of how strongly each input of a network moves its output, found from the gradient of the output with respect to the inputs.

#flashcards/ml/reverse #flashcards/topic/neural-nets Gradient times input:::The slope of the output with respect to an input, multiplied by that input's value: a signed amount of the score credited to the input.

#flashcards/ml/concept #flashcards/topic/neural-nets What are the steps of making a saliency map?
?
Take one input and the one output you care about; ask how the output changes if each input is nudged; let backpropagation answer for all inputs in one backward pass; take the size of each answer and show it; average over many inputs for a general picture; check it with another method.

#flashcards/ml/formula #flashcards/topic/neural-nets What is the saliency of input $j$?
?
$$S_j(x)=\frac{\partial Q(x)}{\partial x_j}$$
- $Q(x)$: the single output being explained (for a DQN, the score of the best move). $x_j$: the $j$-th input.
- $\partial Q/\partial x_j$: how fast $Q$ changes when only $x_j$ moves; like the gradient for weights, but with respect to inputs.
- Matters: it is local, the slope at this $x$ only. A value of 0 can mean "flat here" and not "unimportant".

#flashcards/ml/example #flashcards/topic/neural-nets $Q(x)=2x_1-x_2+0.5x_3$ at $x=(1,3,0)$: what are the saliencies, and what does raising $x_2$ by 0.1 do?::$S=(2,-1,0.5)$. Raising $x_2$ by 0.1 changes $Q$ by about $-1\times0.1=-0.1$. Input 1 is the strongest lever.

#flashcards/ml/formula #flashcards/topic/neural-nets What is gradient times input?
?
$$A_j(x)=x_j\cdot\frac{\partial Q(x)}{\partial x_j}$$
- $x_j$: the input's current value. $\partial Q/\partial x_j$: its slope.
- $A_j$: a signed amount of the score credited to input $j$.
- Matters: an input that is 0 gets no credit however steep its slope. With no constant term in the score, the attributions add up to the score.

#flashcards/ml/example #flashcards/topic/neural-nets $Q(x)=2x_1-x_2+0.5x_3$ at $x=(1,3,0)$: what is gradient times input and what does it sum to?::$A=(1\times2,\;3\times(-1),\;0\times0.5)=(2,-3,0)$. The sum is $-1$, and $Q(x)=2-3+0=-1$. Input 2 pulls the score down by 3; input 3 is off.

#flashcards/ml/concept #flashcards/topic/neural-nets What did the saliency toy show?
?
A small tanh network gave the same slopes by backpropagation and by nudging each input by ±0.001 (−0.1581, −0.1087, 0.0711), so one backward pass replaces one run per input. A function flat until x = 1 had gradient 0 at x = 0 although its output rises to 2 at x = 3: zero gradient does not prove an input is unimportant.

#flashcards/ml/concept #flashcards/topic/neural-nets What did gradient saliency show on the five Snake DQNs?
?
The mean size of the slope was largest for "danger ahead" (0.423) and smallest for the heading inputs (0.22 to 0.24), but the rankings of the 11 inputs agreed only weakly between five networks (mean Spearman correlation 0.46, lowest −0.01).

#flashcards/ml/concept #flashcards/topic/neural-nets Why is a large saliency not the same as importance?
?
It says the output could change a lot if the input changed a little, not that the input drives the decision. It depends on the input's scale and is local to one state; a one-hot input is 0 or 1 and never "a little changed".

#flashcards/ml/concept #flashcards/topic/neural-nets Which facts in the saliency note are from memory?
?
The 2013 to 2014 history of gradient maps, integrated gradients, SmoothGrad, Grad-CAM, and the claim that attributions add up for some ReLU networks. Not checked.

#flashcards/ml/cloze #flashcards/topic/neural-nets A saliency map uses the ==gradient== of the output with respect to the ==inputs==, which backpropagation gives in one backward pass.
