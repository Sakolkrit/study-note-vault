---
tags: [math, calculus, optimization, constrained-optimization, machine-learning]
status: growing
level: 4
tier: 3
---

# Lagrange Multipliers

Lagrange multipliers are a trick for **finding the best value of something while obeying a rule**. They turn "best, but only on this path" into a set of equations you can solve.

## 🟢 1. Simple (age 12)
You want to reach the highest point on a hill, but you must stay on a hiking trail. You can no longer go straight up the steepest slope. At the best spot on the trail, the trail itself runs along a flat line of the hill: if the trail still went up or down, you could walk a bit further and do better. The multiplier is the number that connects "which way the hill is steepest" with "which way the trail's rule pushes".

**Example:** You have 20 metres of fence and want a rectangle with the biggest area. The rule is that the two side lengths add up to 10 (half of the fence). The answer is a square, 5 metres by 5 metres, with area 25. A long thin rectangle like 9 by 1 only has area 9.

## 🔵 2. How it works
1. You have something to make as big (or small) as possible, the **goal** $f$. You also have a **rule** that must hold exactly, written as $g=c$. See [[Calculus]].
2. Picture the goal as a hill and the rule as a trail drawn on the ground. You may only walk on the trail.
3. At the best spot on the trail, you cannot gain by walking along the trail either way. That means the trail runs along a **level line** of the hill there.
4. The direction of steepest climb (the [[Gradient Descent]] direction, called the gradient) always points across a level line. The rule has its own gradient, which also points across the trail. So at the best spot **the two gradients point along the same line**: one is a multiple of the other.
5. That multiple is the **Lagrange multiplier**, called $\lambda$ ("lambda").
6. Write this down as equations, add the rule itself, and solve. You get the best point and $\lambda$ together.
7. $\lambda$ also has a meaning: it is the **price of the rule**. It says how much better the best value would become if the rule were relaxed by one small step.
8. The same trick is used inside [[Support Vector Machine]], [[PCA]] and in finding "the most random" distribution with a given average ([[Entropy]]).

![[lagrange-constraint-curve.svg]]
*A surface (the "hill") with a closed loop drawn on it: the rule. The crosses mark the best and worst points along that loop, with their coordinates. Below, the same loop and some level lines of the surface are drawn on the floor. Image: Jacobmelgaard (original) and Zerodamage (vector), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Lagrange_simple.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **The Lagrangian:** a new function $\mathcal L(x,\lambda)$ that glues the goal and the rule together. Setting all its slopes (with respect to $x$ and to $\lambda$) to zero gives exactly the conditions above.
- **Why the gradients line up:** if the gradient of $f$ had a part along the trail, a small step along the trail would improve $f$. So at the best point the gradient of $f$ must be across the trail, which is the direction of the gradient of $g$.
- **Several rules:** use one multiplier per rule.
- **Rules of the form "at most":** these need the Karush-Kuhn-Tucker (KKT) conditions: the multiplier must be at least 0, and it is 0 for any rule that is not tight. See [[Convex Optimization]].
- **Uses in machine learning:**
  - The [[Support Vector Machine]]'s widest-margin problem is solved through its "dual" form, which comes from these multipliers.
  - [[PCA]]: "find the direction with the most variance, with length 1" is a constrained problem whose answer is an [[Eigenvalue]] problem, and $\lambda$ is the eigenvalue.
  - Maximum entropy ([[Entropy]]): the most spread-out distribution with a fixed average has the exponential form (a standard result; from memory).
- **Weak spots:**
  - It finds **candidate** points (the slopes line up). Those can be a maximum, a minimum or a saddle, so you must check.
  - It needs smooth $f$ and $g$, and the rule's gradient must not be zero at the point.
  - For hard (non-convex) problems it does not give the global best.

```python
import numpy as np
from scipy.optimize import minimize

# maximise f = x*y while staying on the line x + y = 10
res = minimize(lambda p: -p[0] * p[1], [1, 9], constraints={"type": "eq", "fun": lambda p: p[0] + p[1] - 10})
x, y = res.x
print(res.x.round(4), round(-res.fun, 4))           # [5. 5.] 25.0
print("grad f =", (y, x), " grad g = (1, 1)  ->  lambda =", round(y, 4))

best = lambda c: (c / 2) ** 2                       # best value of x*y when x + y = c
print(round((best(10.01) - best(10)) / 0.01, 4))    # how fast the best value grows with c: about lambda

# n numbers adding to 9, maximise their product: Lagrange says all equal 9/n; a grid search needs 100^(n-1) tries
for n in (3, 5):
    print(n, 9 / n, (9 / n) ** n, 100 ** (n - 1))
```

Output: `[5. 5.] 25.0`, then the two gradients are about $(5,5)$ and $(1,1)$, so $\lambda=5$. The next line, `5.0025`, is how fast the best value grows when the rule is loosened from 10 to 10.01: close to $\lambda=5$. The last lines show that for 3 numbers adding to 9 the best product is $3\times3\times3=27$, and for 5 numbers it is $(1.8)^5\approx18.8957$. A grid search with 100 values per free variable would need 10,000 and 100,000,000 tries.

## 🔴 4. Deep / formal
**The recipe.** To make $f(x)$ as large as possible subject to $g(x)=c$, build the Lagrangian and set its slopes to zero:

$$
\mathcal L(x,\lambda)=f(x)-\lambda\big(g(x)-c\big),\qquad \nabla_x\mathcal L=0\;\Longrightarrow\;\nabla f(x)=\lambda\,\nabla g(x),\quad g(x)=c
$$

**Where:**
- $x$ is the list of unknowns. $f$ is the goal, $g$ is the rule's function and $c$ is the value the rule must equal.
- $\lambda$ is the multiplier, a new unknown.
- $\nabla f$ and $\nabla g$ are the gradients (lists of slopes) of the goal and the rule.
- $\mathcal L$ is the Lagrangian. Its slope with respect to $\lambda$ is $-(g-c)$, so setting it to zero gives back the rule.
- **Matters:** without $\lambda$ you would just set $\nabla f=0$ and find the top of the hill, which is probably off the trail and breaks the rule. The multiplier is what lets the gradient of $f$ be non-zero, as long as it is balanced by the rule's gradient.

**Example (tiny numbers):** $f=xy$ and the rule $x+y=10$, so $g=x+y$ and $c=10$.
- Gradients: $\nabla f=(y,\,x)$ and $\nabla g=(1,\,1)$.
- Lining them up: $(y,\,x)=\lambda\,(1,\,1)$, so $y=\lambda$ and $x=\lambda$. Hence $x=y$.
- The rule: $x+y=10$, so $x=y=\mathbf{5}$ and $\lambda=\mathbf{5}$.
- Value: $f=5\times5=\mathbf{25}$.
- Meaning: the best rectangle is a square, and the multiplier is 5 (see the next equation for what that 5 means).

**What $\lambda$ means.** Let $f^*(c)$ be the best value of the goal when the rule is $g=c$. Then:

$$
\lambda=\frac{d f^{*}}{d c}
$$

**Where:**
- $f^{*}(c)$ is the best achievable value of $f$ for a given $c$.
- $c$ is the value of the rule (for example the amount of fence).
- $\lambda$ is the slope of that best value as $c$ changes.
- **Matters:** it is the "price" of the rule, the benefit of loosening it by one unit. If $\lambda$ is 0 the rule is not hurting you. A big $\lambda$ says the rule is expensive, so loosening it helps a lot. (This holds under the usual smoothness conditions; from memory, textbook result.)

**Example (tiny numbers):** for $f=xy$ and $x+y=c$ the best is $x=y=c/2$, so $f^*(c)=c^2/4$.
- Slope: $df^*/dc=c/2$.
- At $c=10$: $10/2=\mathbf{5}$, equal to $\lambda=5$.
- Check by nudging: $f^*(10.01)=10.01^2/4=25.050025$, so the rise over $0.01$ is $0.050025/0.01=5.0025$ (the code printed this).
- Meaning: one extra metre in the sum would add about 5 square metres to the best area.

**Further notes:**
- **Saddle point:** the solution is a saddle of $\mathcal L$ (a maximum in $x$ and, for the dual view, a minimum in $\lambda$). This is the start of Lagrangian duality, used for SVMs (from memory).
- **Inequality rules** and the KKT conditions are in [[Convex Optimization]].
- **Alternatives:** a penalty term added to the loss ([[Regularization]]) enforces a rule softly rather than exactly.

**What came before, and how this improved on it:**
- **Before:** substitution: solve the rule for one unknown, plug it into the goal, and take an ordinary derivative.
- **Its limit:** it needs the rule to be solvable for one variable. With many unknowns or messy rules it becomes long or impossible, and it treats the variables unevenly.
- **What changed:** keep every variable and add one multiplier per rule, so the rule is treated like any other equation. Lagrange introduced the method in his *Mécanique analytique* (1788; from memory, not checked).
- **How much better:** in the code above, for $n$ numbers adding to 9 the multiplier equations give "all equal $9/n$" in one step. A grid search with 100 values per free variable needs 10,000 evaluations for $n=3$ and 100,000,000 for $n=5$ (my arithmetic), and it only gives an approximation.

## Related
- [[Calculus]] · [[Convex Optimization]] · [[Gradient Descent]] · [[Support Vector Machine]] · [[PCA]] · [[Eigenvalue]] · [[Entropy]] · [[Regularization]] · [[Hessian]] · [[Jacobian]] · [[Loss Function]]

## Sources
- Boyd & Vandenberghe, *Convex Optimization*, chapter 5 (free online; from memory)
- Bishop, *Pattern Recognition and Machine Learning*, appendix E (from memory)
- Image: Jacobmelgaard and Zerodamage, Wikimedia Commons, CC BY-SA 3.0 (see the caption in section 2)
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/calculus-optimization Lagrange multiplier:::The number that connects the goal's gradient to the rule's gradient at the best point, and the price of the rule.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the Lagrangian recipe for a constrained maximum and explain each symbol.
?
$$\mathcal L(x,\lambda)=f(x)-\lambda\big(g(x)-c\big),\qquad \nabla_x\mathcal L=0\;\Longrightarrow\;\nabla f(x)=\lambda\,\nabla g(x),\quad g(x)=c$$
- $x$: the unknowns. $f$: the goal. $g$: the rule's function. $c$: the value the rule must equal.
- $\lambda$: the multiplier, a new unknown.
- $\nabla f,\nabla g$: the gradients of the goal and the rule.
- The slope of $\mathcal L$ with respect to $\lambda$ gives back the rule $g=c$.
- Without $\lambda$ you would set $\nabla f=0$ and find the top of the hill, probably off the trail.

#flashcards/math/example #flashcards/topic/calculus-optimization Maximise $f=xy$ with $x+y=10$ using Lagrange multipliers.::Gradients: (y, x) = λ(1, 1), so x = y. The rule gives x = y = 5 and λ = 5. f = 25. The best rectangle is a square.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write what the Lagrange multiplier means and explain each symbol.
?
$$\lambda=\frac{d f^{*}}{d c}$$
- $f^{*}(c)$: the best value of the goal for a given rule value $c$.
- $c$: the value of the rule (for example the amount of fence).
- $\lambda$: the slope of that best value as $c$ changes.
- It is the price of the rule. If $\lambda$ is 0 the rule is not hurting. A big $\lambda$ means loosening the rule helps a lot.

#flashcards/math/example #flashcards/topic/calculus-optimization For $f=xy$ and $x+y=c$, show that $\lambda=df^*/dc$ at $c=10$.::The best is x = y = c/2, so f*(c) = c²/4 and the slope is c/2 = 5 at c = 10. This equals λ = 5. Nudging: f*(10.01) = 25.050025, so the rise over 0.01 is 5.0025 (my run).

#flashcards/math/image #flashcards/topic/calculus-optimization ![[lagrange-constraint-curve.svg]]
What does this picture show?
?
A surface (the hill) with a closed loop drawn on it (the rule). The crosses mark the best and worst points along the loop, with their coordinates. Below, the same loop and level lines of the surface are drawn on the floor.

#flashcards/math/cloze #flashcards/topic/calculus-optimization At the best point on the trail, the gradient of the goal is a ==multiple== of the gradient of the ==rule==.

#flashcards/math/concept #flashcards/topic/calculus-optimization Name two uses of Lagrange multipliers in machine learning.
?
The SVM's widest-margin problem is solved through its dual form. PCA ("most variance, direction of length 1") becomes an eigenvalue problem where the multiplier is the eigenvalue. Maximum entropy with a fixed average is another.

#flashcards/math/concept #flashcards/topic/calculus-optimization Name two weak spots of the method.
?
It finds candidate points only (a maximum, minimum or saddle), so you must check which. It needs smooth functions, and the rule's gradient must not be zero there. For non-convex problems it does not give the global best.

#flashcards/math/example #flashcards/topic/calculus-optimization For 3 numbers adding to 9, what is the best product, and how many tries would a grid with 100 values per free variable need?::All equal 9/3 = 3, so the product is 27. The grid needs 100^2 = 10,000 tries. For 5 numbers: 1.8 each, product 18.8957, and 100^4 = 100,000,000 tries. My arithmetic.
