---
tags: [continual-learning, neural-network, training, evaluation, forgetting, LLM]
status: growing
level: 4
tier: 3
---

# Continual Learning

Teaching a model **many things one after another**, without going back to retrain on everything, and without it forgetting the early lessons. It is the setting in which [[Catastrophic Forgetting]] is the main enemy.

## 🟢 1. Simple (age 12)
Think of a student who takes a new class every term and never gets to re-read old books. A good student keeps what they learned in term one, still learns new things in term five, and even finds that old lessons help with new ones. A bad student learns each new class well, but forgets the earlier ones.

**Example:** A phone keyboard learns your new slang this month. Continual learning is the problem of doing that without forgetting how you spelled things last year, and without storing every message you ever typed.

## 🔵 2. How it works
1. **The setting.** Data arrives as a sequence of tasks (task 1, then task 2, then task 3). After each one, the model is tested on **all** tasks, old and new.
2. **The problem.** A [[Neural Network]] keeps all it knows in one set of shared weights. Training on task 3 can move weights that task 1 needed ([[Catastrophic Forgetting]]).
3. **Two things to balance.** *Stability* is keeping old skills. *Plasticity* is learning new ones. Too much stability and the model learns nothing new. Too much plasticity and it forgets.
4. **Three common families of fixes** (the group names are from memory):
   - **Replay:** keep a few old examples and mix them into new training.
   - **Protect the weights:** add a penalty for moving weights that mattered before.
   - **Give new tasks their own room:** add new weights or a small add-on for each task ([[LoRA]] is an example of a small add-on).
5. **Measuring it.** Put the test results in a table: row = after learning task i, column = score on task j. Three simple averages come out of it: how good the model is at the end, how much it forgot, and whether old lessons helped new tasks.
6. **A warning from the numbers.** Two methods can have the same final average and still differ in how much they forgot. That is why forgetting is measured on its own. The code in level 3 shows this.
7. In large language models, the same problem appears when a model is fine-tuned ([[Fine-tuning]]) or trained further on new text. Most methods above are tested on small image tasks, not on LLMs.

## 🟠 3. Technical detail
- **Task-incremental vs other settings.** In the survey of De Lange et al. (2019) the task boundaries are clear and the model knows which task each example belongs to (abstract read). Real data often has no such boundaries; that harder case is not covered here.
- **Replay.** Needs memory for old examples and may not be allowed for private data. More stored data forgets less (code below). Gradient Episodic Memory (Lopez-Paz and Ranzato, 2017) keeps some old examples and uses them to stop updates that would hurt old tasks (abstract and section 2 read; the algorithm itself was not read).
- **Penalties.** Pull the weights towards their old values, more strongly for important ones. See the equation in [[Catastrophic Forgetting]].
- **Isolation.** Per-task add-ons grow with the number of tasks, but cannot overwrite each other.
- **Capacity.** A model that is too small cannot hold all tasks at once. The code shows this: it has only 6 weights, so a single setting cannot fit three rules.
- **Related ideas.** [[Transfer Learning]] is one move from a trained model to a new task; continual learning is many moves in a row.
- **The code below is a toy.** Three tasks, each a random rule on 6 numbers, one 6-weight model. It builds the results table, computes the three averages, and compares plain one-by-one training with replay and a pull to the old weights, plus training on all data at once as a reference. One random seed.

```python
import numpy as np

rng = np.random.RandomState(0)
d, T = 6, 3
U = rng.randn(T, d)                             # task k: label is 1 when x . U[k] > 0 (three different rules)


def data(k, n):
    X = rng.randn(n, d)
    return X, (X @ U[k] > 0).astype(float)


train = [data(k, 1000) for k in range(T)]
test = [data(k, 2000) for k in range(T)]


def acc(w, k):
    X, y = test[k]
    return float((((X @ w) > 0) == (y > 0)).mean())


def fit(w, X, y, steps=400, lr=0.5, anchor=None, lam=0.0):
    w = w.copy()
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y)
        if anchor is not None:
            g += lam * (w - anchor)
        w -= lr * g
    return w


def run(name, replay=0, lam=0.0):
    w, R, buf = np.zeros(d), np.zeros((T, T)), []
    for t in range(T):                          # tasks arrive one after another
        X, y = train[t]
        if replay and buf:
            X = np.vstack([X] + [b[0] for b in buf]); y = np.concatenate([y] + [b[1] for b in buf])
        w = fit(w, X, y, anchor=w if lam else None, lam=lam)
        buf.append((train[t][0][:replay], train[t][1][:replay]))
        R[t] = [acc(w, k) for k in range(T)]    # R[i, j] = accuracy on task j after learning task i
    ACC = R[T - 1].mean()
    BWT = np.mean([R[T - 1, i] - R[i, i] for i in range(T - 1)])
    FWT = np.mean([R[i - 1, i] - 0.5 for i in range(1, T)])    # 0.5 = chance before any training (balanced labels)
    print(f"{name:22s} ACC {ACC:.3f}  BWT {BWT:+.3f}  FWT {FWT:+.3f}  final row {R[T - 1].round(2)}")
    return R


R = run("naive, one by one")
print("   full R matrix of the naive run (row = after task i, column = task j):\n", R.round(2))
run("replay 100 per task", replay=100)
run("replay 500 per task", replay=500)
run("pull to old weights .2", lam=0.2)
wj = fit(np.zeros(d), np.vstack([t[0] for t in train]), np.concatenate([t[1] for t in train]))
print(f"{'all data together':22s} ACC {np.mean([acc(wj, k) for k in range(T)]):.3f}   (reference: no sequence)")
```

What it printed: the naive run has final accuracies (task 1, 2, 3) of (0.82, 0.57, 1.00), ACC 0.796, BWT −0.298. Replay of 100 per task: ACC 0.800, BWT −0.267. Replay of 500: ACC 0.798, BWT −0.178. Pull to old weights: ACC 0.791, BWT −0.180. All data together: ACC 0.793. So **all methods ended with about the same average, but forgetting (BWT) was 40% smaller with 500 replayed examples** (−0.178 against −0.298). The 0.01 differences in ACC are within what one seed could give, so I do not read anything into them.

## 🔴 4. Deep / formal
Let $R_{i,j}$ be the test accuracy on task $j$ right after learning task $i$ (the table from level 2), with $T$ tasks. Two summary numbers (the metrics of Lopez-Paz and Ranzato, 2017; I read their definitions):

$$
\begin{aligned}
\mathrm{ACC}&=\frac1T\sum_{i=1}^{T}R_{T,i}\\
\mathrm{BWT}&=\frac1{T-1}\sum_{i=1}^{T-1}\bigl(R_{T,i}-R_{i,i}\bigr)
\end{aligned}
$$

**Where:**
- $T$ is the number of tasks and $R_{T,i}$ the accuracy on task $i$ **at the very end** (after the last task).
- $R_{i,i}$ is the accuracy on task $i$ right after it was learned, so $R_{T,i}-R_{i,i}$ is how much was lost (or gained) since then.
- ACC is the average final accuracy. BWT ("backward transfer") is the average change on earlier tasks, over the first $T-1$ tasks. It is negative when the model forgets.
- **Matters:** ACC alone can look fine while the model forgot a lot and made up for it by learning the last task very well. BWT isolates the forgetting. The authors say that when two models have similar ACC, the one with larger BWT is preferable. (The paper also defines a forward-transfer number, FWT: how much earlier tasks helped a task before it was trained.)

**Example (tiny numbers):** three tasks, with the table below (rows = after task 1, 2, 3).

| after \ task | 1 | 2 | 3 |
|---|---|---|---|
| 1 | 0.9 | 0.5 | 0.5 |
| 2 | 0.6 | 0.9 | 0.5 |
| 3 | 0.5 | 0.6 | 0.9 |

- ACC $=(0.5+0.6+0.9)/3=\mathbf{0.667}$.
- BWT $=\bigl((0.5-0.9)+(0.6-0.9)\bigr)/2=\mathbf{-0.35}$.
- Meaning: on average the model ended 0.35 below where it was right after learning each old task, so it forgot a lot. If it had kept everything, the final row would read 0.9, 0.9, 0.9 and BWT would be 0.

**Further notes (from the code above):**
- **Same ACC, different forgetting.** ACC was 0.796 to 0.800 for the three variants without a penalty, 0.791 for the penalty, and 0.793 for training on all the data together, yet BWT went from −0.298 to −0.178 and −0.180. This is the reason to report both.
- **Replay needs room.** 100 old examples per task changed BWT only from −0.298 to −0.267. 500 per task gave −0.178.
- **Costs.** Replay stores data. The penalty stores one old copy of the weights. Neither helped the final score here, because 6 weights cannot satisfy three rules at once, so the best possible average is limited by capacity.
- **Forward transfer was positive in every run** (+0.068 for the naive run) because the random rules happen to point partly in the same direction. That is an accident of the toy.
- **What the toy cannot show:** results on real images, text or LLMs, task orders that matter, or tasks with no boundaries. One seed was used, so small gaps are noise.

**What came before, and how this improved on it:**
- **Before:** ordinary supervised learning, which assumes one fixed task and examples drawn from one fixed distribution (as Lopez-Paz and Ranzato state in the introduction; read).
- **Its limit:** trained one task after another, the model forgets the old ones, and there was no agreed way to measure that (their abstract: they "propose a set of metrics").
- **What changed:** continual learning became a defined setting: tasks streamed in order, a result table, and the ACC, BWT and FWT averages (Lopez-Paz and Ranzato, arXiv 1706.08840, 2017). Surveys later sorted the methods and added a way to measure the stability-plasticity trade (De Lange et al., arXiv 1909.08383, 2019; abstract read, which compares 11 methods and 4 baselines on three benchmarks).
- **How much better:** in this note's toy, 500 replayed examples per task made BWT −0.178 against −0.298 for plain one-by-one training, at about the same ACC. I did not read the published comparison tables, so I give no published figure.

## Related
- [[Catastrophic Forgetting]] · [[Fine-tuning]] · [[Transfer Learning]] · [[LoRA]] · [[Neural Network]] · [[Pre-training]] · [[Evaluation Metrics]] · [[Overfitting]] · [[Regularization]] · [[Instruction Tuning]] · [[LLM]] · [[Fisher Information]]

## Sources
- Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual Learning*, arXiv 1706.08840 (the PDF text version dated 13 Sep 2022, v6). Abstract, introduction and the metrics section were read; the method section and experiments were not.
- De Lange, Aljundi, Masana, Parisot, Jia, Leonardis, Slabaugh, Tuytelaars, *A continual learning survey: Defying forgetting in classification tasks*, arXiv 1909.08383 (first submitted 18 Sep 2019). **Abstract only.**
- The three family names (replay, penalty, isolation) are a common grouping **from memory**.
- Image: none. Two Wikimedia Commons searches found only unrelated files and one business slide with unsourced percentages (not used, since the numbers have no source).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Continual learning:::Learning a sequence of tasks one after another, keeping what was learned before, still learning new things, and ideally using old lessons to help the new ones.

#flashcards/ml/formula #flashcards/topic/training Write the two continual-learning scores ACC and BWT and explain each symbol.
?
$$\begin{aligned} \mathrm{ACC}&=\frac1T\sum_{i=1}^{T}R_{T,i}\\ \mathrm{BWT}&=\frac1{T-1}\sum_{i=1}^{T-1}\bigl(R_{T,i}-R_{i,i}\bigr) \end{aligned}$$
- $T$: the number of tasks. $R_{T,i}$: the accuracy on task $i$ at the very end (after the last task).
- $R_{i,i}$: the accuracy on task $i$ right after it was learned, so $R_{T,i}-R_{i,i}$ is how much was lost (or gained) since.
- ACC: the average final accuracy. BWT (backward transfer): the average change on earlier tasks, over the first $T-1$ tasks. Negative when the model forgets.
- ACC alone can look fine while the model forgot a lot and made up for it by learning the last task very well. BWT isolates the forgetting.

#flashcards/ml/example #flashcards/topic/training Final rows: after task 1: (0.9, 0.5, 0.5), after task 2: (0.6, 0.9, 0.5), after task 3: (0.5, 0.6, 0.9). What are ACC and BWT?::ACC $=(0.5+0.6+0.9)/3=0.667$. BWT $=((0.5-0.9)+(0.6-0.9))/2=-0.35$. On average the model ended 0.35 below where it was right after learning each old task. If it kept everything, the final row would be 0.9, 0.9, 0.9 and BWT 0.

#flashcards/ml/concept #flashcards/topic/training Why report both ACC and BWT?::Models can have similar ACC but very different forgetting. In the note's code ACC was about 0.79 to 0.80 for all variants while BWT ranged from −0.298 to −0.178.

#flashcards/ml/reverse #flashcards/topic/training Replay:::Store some old examples and train on them again. In the note's code 100 per task changed BWT only from −0.298 to −0.267, and 500 per task gave −0.178.
