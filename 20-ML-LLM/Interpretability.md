---
tags: [interpretability, explainability, XAI, neural-network, evaluation, ML]
status: growing
level: 4
tier: 3
---

# Interpretability

Interpretability is **the effort to understand why a trained model gives the answers it gives**. A model that works can still be a black box: you see what goes in and what comes out, but not the reasons. Interpretability tools try to open the box a little, or to test it from outside.

## 🟢 1. Simple (age 12)
Your friend is great at chess but cannot explain their moves: "it just feels right". If you want to learn from them, or check that they are not cheating, you need a way to find out what they are looking at. You could **ask them to cover one part of the board and see whether their move changes**, or **watch where their eyes go**. Interpretability does the same to a model: it covers inputs, watches where the model is sensitive, and then asks the hard question: does the explanation really match what the model does?

**Example:** a robot that plays Snake wins most games. You cover its "wall ahead" sensor and it starts crashing, so it uses that sensor. You cover the "heading" sensor and it hardly changes. Now you know something about how it thinks.

## 🔵 2. How it works
1. **Choose the question.** Which inputs matter overall (global)? Why this one answer (local)? What does the model learn inside?
2. **Pick a tool that fits the question:**
   - **Take inputs away or scramble them** and watch the answer: [[Permutation Importance]].
   - **Ask the maths which inputs the output is sensitive to:** [[Saliency Map]].
   - **Compare with something simple** that you do understand, such as a hand-written rule or a lookup table.
   - **Look inside** at the learned numbers and activations (not done in my projects).
3. **Run it on many cases**, not one, and on more than one trained copy of the model.
4. **Test the explanation.** If the tool says "input A matters", change input A and see whether the behaviour changes as predicted.
5. **Say what it does not show.** Most tools describe sensitivity or use, not cause.

*I found no free, on-topic overview picture of interpretability on Wikimedia Commons (the results were papers, a talk transcript, and a chart for a spam-bot model I used in [[Permutation Importance]]), so this note has no picture.*

## 🟠 3. Technical detail
- **Intrinsic or post-hoc.** Some models are readable by design (a short decision tree, a linear model with few inputs); you read the model itself. Others are trained first and explained afterwards ("post-hoc"), which is what the tools here do for a neural network.
- **Global or local.** A *global* explanation describes the model's overall behaviour ("danger ahead matters most"); a *local* one explains one decision ("it turned left because the food was left").
- **Model-agnostic or model-specific.** [[Permutation Importance]] needs only the model's predictions. A [[Saliency Map]] needs the gradients inside a neural network.
- **Faithful or only plausible.** An explanation is *faithful* if it describes what the model really uses, and merely *plausible* if it just looks reasonable to a person. A faithful tool can still disappoint a person, and a plausible one can mislead.
- **Code: a danger in scrambling.** [[Permutation Importance]] shuffles one input column, but in Snake the four "heading" inputs form a one-hot group (exactly one is 1). The code observes 20,000 steps of a random player, shuffles only the "heading up" column and counts how many rows stop being a valid heading. It needs `snake.py` from [[Snake DQN]] in the same folder:

```python
# file: toy_invalid.py
import random
import numpy as np
from snake import Snake

env, rng, obs = Snake(), random.Random(0), []
o = env.reset(seed=0)
for _ in range(20000):                                                # observations from a random player
    obs.append(o); o, _, done = env.step(rng.randrange(3))
    if done: o = env.reset()
X = np.array(obs); p = float(X[:, 3].mean())                          # column 3 = "heading up" (columns 3 to 6 are one-hot)
Xs = X.copy(); Xs[:, 3] = X[np.random.default_rng(0).permutation(len(X)), 3]    # permutation importance: shuffle that one column
bad = float((Xs[:, 3:7].sum(axis=1) != 1).mean())
print(f"share of rows with heading up = 1: {p:.3f}")
print(f"after shuffling that column, rows whose heading is no longer exactly one direction: {bad:.3f}")
print(f"prediction 2 p (1 - p) = {2 * p * (1 - p):.3f}")
```

```text
share of rows with heading up = 1: 0.238
after shuffling that column, rows whose heading is no longer exactly one direction: 0.366
prediction 2 p (1 - p) = 0.363
```

- **Reading it:** "heading up" is 1 in 23.8% of rows. After shuffling that one column, 36.6% of the rows no longer have exactly one heading (a row now has none or two), close to my prediction $2p(1-p)=0.363$: a row breaks when the old value is 1 and the new one 0, or the other way round. So **more than a third of the scrambled states are states that cannot happen**, and a model's reaction to them may say little about how it plays the real game.
- **What I did on the Snake DQN ([[Looking Inside Snake DQN]]):**
  - Shuffled each of the 11 inputs and counted how often the chosen move changed: 8% to 18% for every input, with no clear winner.
  - Froze each input at its average and played 200 games: freezing "danger ahead" cost about 4 foods (from 15.5 to 11.4); freezing "food up" cost anything from almost nothing to about 12 foods depending on the network.
  - Took the gradient of the best move's score: "danger ahead" 0.423, the heading inputs about 0.22 to 0.24.
  - Compared five networks trained the same way: the rankings of the inputs agreed only moderately (Spearman 0.37 to 0.51 for the three measures).
  - Compared with the hand-written rule: the network picked the same move as the rule in 93.6% of the visited states, and never stepped into an immediate danger when a safe move existed.
  - Trained a plain table on the same 11 inputs: with the same 60,000 steps it scored 13.67 against 15.46 for the network, and with 300,000 steps 15.84.

## 🔴 4. Deep / formal
- **No single right answer.** Different tools answer different questions and can disagree. In my project the three measures ranked the 11 inputs differently, and two networks trained with different random seeds ranked them differently too. That is the usual picture, not a bug in one tool.
- **Correlated and linked inputs.** Inputs that carry the same information share credit ([[Permutation Importance]]), and linked inputs (one-hot groups) make scrambling unreliable (above).
- **Explanations are not causes.** They describe what a fitted model uses. A model may use a proxy, and the explanation then points at the proxy.
- **Evaluating an explanation is hard.** One can ask: does removing the "important" inputs hurt more than removing random ones? Do the explanations stay the same across retrains? Do people make better predictions of the model after seeing them? (From memory, as general ideas; I tested only the first two in a small way.)
- **Inside the network.** Researchers also look at neurons and groups of neurons, probe layers with small classifiers, and try to reverse-engineer circuits (from memory, none of it run in this vault). For large language models this is an active area, often called mechanistic interpretability.
- **Safety and trust.** Interpretability is one input to deciding when to trust a model; it does not replace testing it ([[Evaluation Metrics]], [[Confidence Interval]]).
- **A plain baseline helps.** My tabular Q-learning check shows how useful a simple comparison is: a model that was feared to be necessary (a neural network) was only faster to train than a table of at most 256 rows, not better at the end.

**What came before, and how this improved on it:**
- **Before:** judge a model only by its score on a test set, or by reading a linear model's weights.
- **Its limit:** a score does not say whether the model is right for the right reasons, and weights do not exist in a readable form for a deep network.
- **What changed:** tools that probe a trained model from the outside (scrambling or removing inputs) or from the inside (gradients). Modern work grew in the 2010s with image-network gradient maps and model-agnostic tools such as LIME and SHAP (from memory, not checked).
- **How much better:** measured in [[Looking Inside Snake DQN]]: the tools told me which input a network cannot do without ("danger ahead": about 4 foods lost when frozen) and that the network agrees with a hand-written rule in 93.6% of states, which a score alone would not. They also showed their limits: no stable ranking across five networks, and scrambled states that are invalid. I have no published figure to quote.

## Related
- [[Permutation Importance]] · [[Saliency Map]] · [[Looking Inside Snake DQN]] · [[Snake DQN]] · [[Deep Q-Network]] · [[Neural Network]] · [[Backpropagation]] · [[Evaluation Metrics]] · [[Confidence Interval]] · [[Random Forest]] · [[Attention]] · [[Overfitting]]

## Sources
- My own toy (Python 3.9.6, NumPy 1.24.4) and the project [[Looking Inside Snake DQN]]. The taxonomy (intrinsic and post-hoc, global and local, faithful and plausible), LIME, SHAP, probing and mechanistic interpretability are from memory; I did not read any paper or documentation for this note.
- No image (see level 2).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Interpretability:::The effort to understand why a trained model gives the answers it does, by probing it from outside (scrambling inputs), from inside (gradients, neurons) or by comparing it with something simple.

#flashcards/ml/reverse #flashcards/topic/ml-foundations Faithful explanation:::An explanation that describes what the model really uses, as opposed to one that only looks reasonable to a person.

#flashcards/ml/concept #flashcards/topic/ml-foundations What are the steps of looking into a model?
?
Choose the question (global, local or inside); pick a fitting tool (scramble or remove inputs, gradients, comparison with a simple baseline); run it on many cases and on more than one trained copy; test the explanation by changing the input it names; say what it does not show.

#flashcards/ml/concept #flashcards/topic/ml-foundations What is the difference between global and local explanations, and between intrinsic and post-hoc?
?
Global describes the model's overall behaviour; local explains one decision. Intrinsic means the model is readable by design (a short tree, a linear model); post-hoc means a trained model is explained afterwards.

#flashcards/ml/concept #flashcards/topic/ml-foundations Why is shuffling one input column unsafe for a one-hot group in Snake?
?
The four heading inputs have exactly one 1. "Heading up" is 1 in 23.8% of rows; after shuffling that column 36.6% of rows no longer have exactly one heading (prediction 2p(1-p) = 36.3%). Over a third of the scrambled states cannot happen.

#flashcards/ml/concept #flashcards/topic/ml-foundations What did freezing each input do to the Snake DQN's score?
?
Freezing "danger ahead" at its average cost about 4 foods (15.5 to 11.4); freezing "food up" left a mean score of 11.29, with the five networks between 3.5 and 15.5 (from almost nothing lost to about 12 foods); the heading inputs cost about 0.5 to 1.5 foods. Five networks, 200 games.

#flashcards/ml/concept #flashcards/topic/ml-foundations How much did the Snake DQN copy the hand-written rule?
?
It chose the same move as the rule in 93.6% of the visited states (runs 91.0% to 96.1%), and never stepped into an immediate danger when a safe move existed, yet it scored a little higher than the rule (15.46 against 14.85, a gap that is not firmly established).

#flashcards/ml/concept #flashcards/topic/ml-foundations Did the different explanation tools agree on the five Snake DQNs?
?
Only moderately. Rankings of the 11 inputs agreed between networks with a mean Spearman correlation of 0.51 (shuffling), 0.37 (freezing) and 0.46 (gradient); the lowest pairs were 0.31, 0.07 and −0.01.

#flashcards/ml/concept #flashcards/topic/ml-foundations Was a neural network needed for Snake with 11 inputs?
?
Not for the final level. A plain table (at most 256 rows) scored 13.67 after 60,000 steps against 15.46 for the DQN, but 15.84 after 300,000 steps. The network learned faster; the table was not tuned and there were 5 runs each.

#flashcards/ml/concept #flashcards/topic/ml-foundations Why is an explanation not the same as a cause?
?
Tools describe what a fitted model uses or is sensitive to. A model can lean on a proxy, and then the explanation points at the proxy rather than the real cause.

#flashcards/ml/concept #flashcards/topic/ml-foundations Which facts in the interpretability note are from memory?
?
The taxonomy (intrinsic or post-hoc, global or local, faithful or plausible), LIME, SHAP, probing and mechanistic interpretability, and the 2010s history. Not checked.

#flashcards/ml/cloze #flashcards/topic/ml-foundations A ==faithful== explanation describes what the model really uses, while a ==plausible== one only looks reasonable to a person.
