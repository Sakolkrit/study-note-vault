---
tags: [calculus, derivative, chain-rule, math, optimization, machine-learning]
status: growing
level: 4
tier: 1
---

# Calculus

## 🟢 1. Simple (age 12)
Calculus is the maths of **change**. Its main tool, the **derivative**, tells you **how fast something is changing right now**. It is the speedometer of maths: your car's position changes over time, and the speedometer tells you how quickly at this very moment.

**Example:** You roll a ball down a hill. At the top it moves slowly, and lower down it moves faster. If you stop the clock at any moment and ask "how steep is the hill right here?", the answer is the derivative at that spot. Steep means big, flat means zero, and a negative number means going downhill.

## 🔵 2. How it works
1. Take a curve (a **function**): put in a number, get out a number.
2. To find how steep it is at one point, pick a **second point** a small step $h$ away, and draw a straight line through both points. Its steepness is "rise over run".
3. Now move the second point **closer and closer** to the first. The line swings round and settles into a final line that just touches the curve: the **tangent**.
4. The steepness of the tangent is the **derivative** at that point.
5. **Positive** derivative: going up. **Negative:** going down. **Zero:** flat, which is where a smooth curve has its top or bottom. This is why derivatives find the **lowest point** of a [[Loss Function]].
6. When one thing depends on another which depends on a third (like a chain), the changes **multiply along the chain**. This is the **chain rule**, and it is the heart of [[Backpropagation]].

![[tangent-secant-limit.svg]]
*The tangent as a limit: the red curve C, the fixed point P at $x_0$, and a second point Q at $x_0+h$. The black lines through P and Q are secants. As Q slides toward P, the secants turn and approach the blue tangent line L. Image: Inductiveload, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Tangent_as_Secant_Limit.svg), public domain.*

## 🟠 3. Technical detail
- **Derivative of a function** $f$ at a point: written $f'(x)$ or $\dfrac{df}{dx}$. It is the slope of the tangent line there.
- **Handy rules:**
  - Power: the derivative of $x^n$ is $n\,x^{n-1}$ (so $x^2\to2x$).
  - Sum: add the derivatives. Constant times a function: keep the constant.
  - Product and chain rules (below).
  - $e^x$ stays $e^x$; $\ln x$ gives $1/x$. (These two appear in [[Softmax]] and cross-entropy.)
- **Partial derivative:** for a function of many inputs, change one input and hold the rest fixed. A model has billions of weights, so the loss has billions of partial derivatives.
- **Gradient:** the [[Vector]] of all partial derivatives. It points in the direction the function climbs fastest. [[Gradient Descent]] walks the opposite way, downhill.
- **Integral:** the opposite of a derivative, measuring accumulated amount (area under a curve). It is how probabilities of continuous values are found ([[Probability]]).
- **Numerical derivative:** computers can estimate a derivative by using a tiny $h$ in the definition. Libraries instead compute exact derivatives automatically using the chain rule ([[Backpropagation]]).

```python
f = lambda x: x ** 2
for h in (1, 0.1, 0.01, 0.001):
    print(h, round((f(3 + h) - f(3)) / h, 3))   # 7.0, 6.1, 6.01, 6.001  -> 6

g = lambda x: 3 * x + 1
F = lambda x: g(x) ** 2                           # f(g(x)) with f(u) = u**2
h = 1e-6
print(round((F(2 + h) - F(2 - h)) / (2 * h), 3))  # 42.0  (chain rule: 14 * 3)
```

## 🔴 4. Deep / formal
**The derivative as a limit.** The slope of the secant through $x$ and $x+h$, as the gap $h$ shrinks to zero:

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

**Where:**
- $f$ is the function and $x$ is the point where you want the slope. $f'(x)$ is the answer: the instantaneous rate of change.
- $h$ is the small step to the second point. **Matters:** it is the horizontal gap between P and Q in the picture. It must be non-zero in the fraction (you cannot divide by 0), so we look at what happens as $h$ gets closer and closer to 0.
- $f(x+h)-f(x)$ is the **rise**: how much the output changed over that step. Dividing by $h$ (the **run**) gives rise over run.
- $\lim_{h\to0}$ means "the value this fraction settles towards as $h$ shrinks". **Matters:** this limit is what turns the secant into the tangent. Without it you would only have an average slope over a stretch, not the slope at one point.

**Example (tiny numbers):** $f(x)=x^2$ at $x=3$.
- $f(3)=9$. For step $h$: $\dfrac{f(3+h)-9}{h}=\dfrac{(3+h)^2-9}{h}=\dfrac{6h+h^2}{h}=6+h$.
- $h=1$: slope $=\mathbf{7}$. $h=0.1$: $\mathbf{6.1}$. $h=0.01$: $\mathbf{6.01}$.
- As $h\to0$ the value settles at $\mathbf{6}$. So $f'(3)=6$, which matches the power rule $2x=2\cdot3=6$.
- Meaning: at $x=3$ the curve rises 6 units for each 1 unit step to the right. At $x=0$ the slope would be $0$, the flat bottom of the bowl.

**The chain rule.** If $y=f(u)$ and $u=g(x)$, so that $y=f(g(x))$, then:

$$
\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}=f'\big(g(x)\big)\,g'(x)
$$

**Where:**
- $x$ is the input, $u=g(x)$ is the middle value, and $y=f(u)$ is the final output.
- $\dfrac{du}{dx}$ is how fast the middle value $u$ moves when $x$ moves. $\dfrac{dy}{du}$ is how fast the output moves when $u$ moves.
- Multiplying them gives how fast the output moves when $x$ moves. **Matters:** effects along a chain **multiply**. If $u$ changes 3 times as fast as $x$, and $y$ changes 14 times as fast as $u$, then $y$ changes $3\cdot14=42$ times as fast as $x$.
- $f'(g(x))$ means: take the derivative of the outer function, but evaluate it at the **inner** value $g(x)$, not at $x$. **Matters:** forgetting this is the most common chain-rule mistake.

**Example (tiny numbers):** $g(x)=3x+1$, $f(u)=u^2$, at $x=2$.
- Inner value: $u=g(2)=3\cdot2+1=\mathbf{7}$.
- $\dfrac{du}{dx}=3$. $\dfrac{dy}{du}=2u=2\cdot7=\mathbf{14}$.
- $\dfrac{dy}{dx}=14\cdot3=\mathbf{42}$.
- Check by expanding: $y=(3x+1)^2=9x^2+6x+1$, so $\dfrac{dy}{dx}=18x+6=18\cdot2+6=42$. Same answer.
- Meaning: near $x=2$, nudging $x$ by $0.01$ changes $y$ by about $0.42$. A neural network is a long chain of such steps, which is why the chain rule is the engine of [[Backpropagation]].

**Further notes:**
- **Gradient descent uses it directly:** step $w\leftarrow w-\eta\,\partial L/\partial w$, where the partial derivative comes from the chain rule ([[Gradient Descent]]).
- **Second derivative:** the derivative of the derivative says how the slope itself changes (curvature). It tells a valley bottom (curving up) from a hilltop (curving down).
- **Not every function has a derivative everywhere:** ReLU ([[Neural Network]]) has a corner at 0. In practice a slope of 0 or 1 is simply chosen there.
- **Limits are made precise** with the "epsilon-delta" definition in a real analysis course; the intuitive version above is enough for machine learning.

**What came before, and how this improved on it:**
- **Before:** estimate a slope by measuring between two nearby points.
- **Its limit:** the answer depends on the step size, and a very small step is ruined by rounding error.
- **What changed:** rules for exact derivatives (the power rule and others), invented separately by Newton and Leibniz in the late 1600s (from memory, not checked), so a computer can use them.
- **How much better:** in the code below, the true slope of $x^3$ at 2 is 12. The two-sided estimate has an error of 1.0e+00 with a step of 1, 2.1e-10 with 1e-05, and 1.2e-01 with 1e-14, because the step is then too small for the number format. The rule gives 12 exactly. This is also why [[Backpropagation]] uses exact rules rather than steps.

**Check it yourself:**

```python
f = lambda x: x ** 3                                  # true slope at x = 2 is 3 * 2^2 = 12
print("slope of x^3 at 2: exact 12")
for h in (1.0, 0.1, 1e-5, 1e-10, 1e-14):
    est = (f(2 + h) - f(2 - h)) / (2 * h); print(f"  step h = {h:g}: estimate {est:.10f}, error {abs(est - 12):.1e}")
print("a rule (power rule) gives 12 exactly with no step at all")
```

## Related
- [[Gradient Descent]] · [[Backpropagation]] · [[Loss Function]] · [[Neural Network]] · [[Softmax]] · [[Vector]] · [[Matrix]] · [[Probability]] · [[Linear Regression]] · [[Convex Optimization]] · [[Chain Rule]] · [[Jacobian]]

## Sources
- 3Blue1Brown, *Essence of Calculus* (video series)
- Strang, *Calculus* (free online, MIT OpenCourseWare)
- Stewart, *Calculus* (standard textbook)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/calculus-optimization Derivative:::How fast something is changing right now, the slope of the curve at one single point.

#flashcards/math/concept #flashcards/topic/calculus-optimization What does a derivative that is big, zero or negative mean?::Big means steep. Zero means flat. Negative means going downhill.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the definition of the derivative as a limit and explain each symbol.
?
$$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$
- $f$: the function. $x$: the point where you want the slope. $f'(x)$: the answer, the instantaneous rate of change.
- $h$: the small step to a second point. It must be non-zero in the fraction, so we look at what happens as $h$ gets closer to 0.
- $f(x+h)-f(x)$: the rise (how much the output changed). Dividing by $h$ (the run) gives rise over run.
- $\lim_{h\to0}$: the value the fraction settles towards as $h$ shrinks. It turns the secant into the tangent. Without it you only have an average slope over a stretch.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the chain rule and explain each symbol.
?
$$\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}=f'\big(g(x)\big)\,g'(x)$$
- $x$: the input. $u=g(x)$: the middle value. $y=f(u)$: the final output.
- $dy/du$: how fast the output moves when $u$ moves. $du/dx$: how fast $u$ moves when $x$ moves.
- Multiplying them gives how fast the output moves when $x$ moves. Effects along a chain multiply.
- $f'(g(x))$: the derivative of the outer function, evaluated at the inner value $g(x)$, not at $x$. Forgetting this is the most common mistake.

#flashcards/math/cloze #flashcards/topic/calculus-optimization In the chain rule, effects along a chain ==multiply==, and the outer derivative must be evaluated at the ==inner== value $g(x)$.

#flashcards/math/example #flashcards/topic/calculus-optimization For $f(x)=x^2$, what does the slope fraction give for $h=1$, $0.1$, $0.01$ at $x=3$, and what is $f'(3)$?::7, 6.1, 6.01, settling at $f'(3)=6$ (same as the power rule $2x$).

#flashcards/math/example #flashcards/topic/calculus-optimization For $g(x)=3x+1$ and $f(u)=u^2$ at $x=2$, what is $dy/dx$?::$u=7$, $du/dx=3$, $dy/du=2u=14$, so $dy/dx=14\times3=42$. Check: $y=9x^2+6x+1$, $y'=18x+6=42$.

#flashcards/math/image #flashcards/topic/calculus-optimization ![[tangent-secant-limit.svg]]
What does this picture show?
?
The tangent as a limit. P is fixed, Q is a second point at distance $h$. The lines through P and Q are secants. As Q slides towards P, the secants turn and approach the tangent line.
