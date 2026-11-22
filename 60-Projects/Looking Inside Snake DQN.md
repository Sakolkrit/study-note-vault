---
tags: [project, interpretability, reinforcement-learning, dqn, pytorch, snake, permutation-importance, saliency]
status: done
stack: Python 3.9.6, PyTorch 2.8.0 (CPU only), NumPy 1.24.4, SciPy 1.8.1 (only for rank correlation)
goal: look inside the five trained Snake DQNs to see which inputs they use, whether the tools agree, and whether a plain table could have done the same job
---

# Looking Inside Snake DQN

> **In one line:** take the five trained networks of [[Snake DQN]] and (1) train a plain **table** on the same 11 inputs to see whether a network was needed, then (2) probe the networks with three tools (shuffling, freezing and gradients) to see which inputs they use, and how far the tools and the five networks agree.

## Goal
Done means: (1) a tabular [[Q-Learning]] player on the same game, the same exploration and 5 runs, tested on the same 200 games as the networks, (2) for every one of the 11 inputs of every network, three measures ([[Permutation Importance]] as the share of moves that change, the score when the input is frozen, and the size of the gradient, see [[Saliency Map]]), (3) a check of how much the networks and the measures agree with each other and with the hand-written rule, and (4) a list of what these measures cannot tell. The question was left open at the end of [[Snake DQN]]: "a plain table would likely have done as well; I did not run it".

## Result in short
- **A table needed more steps, but caught up.** Same game, same 11 inputs, same exploration schedule, 5 runs each, mean foods per game on 200 test games:

| Player | Steps | Mean | Spread between runs |
|---|---|---|---|
| tabular Q-learning | 60,000 | 13.67 | 0.64 |
| tabular Q-learning | 300,000 | 15.84 | 0.78 |
| full DQN (from [[Snake DQN]]) | 60,000 | 15.46 | 0.31 |
| hand-written rule | none | 14.85 | none |

  At 60,000 steps the table is well behind the network (1.8 foods, about three to six times the spread between runs of 0.3 to 0.6). At 300,000 steps the table scores 15.84, and its 0.4 gap above the network is smaller than the spread between runs. So here **the network learned faster; it did not end up better**. The table uses 251 to 256 of its at most 256 rows. I did not tune the table (its learning rate is fixed at 0.1), and I added the 300,000-step row **after** seeing the 60,000-step result, so treat it as a follow-up and not as a planned test.
- **Which inputs the networks use** (five networks, 200 games each; the three measures, mean over the networks with the lowest and highest in brackets):

| Input | Moves that change when shuffled | Mean score with the input frozen | Mean size of the gradient |
|---|---|---|---|
| danger ahead | 0.108 (0.100 to 0.114) | 11.41 (10.9 to 11.9) | 0.423 |
| danger right | 0.083 (0.070 to 0.098) | 11.86 (10.7 to 12.5) | 0.310 |
| danger left | 0.094 (0.065 to 0.115) | 12.22 (10.7 to 13.4) | 0.324 |
| heading up | 0.163 (0.115 to 0.192) | 14.93 (13.2 to 16.0) | 0.240 |
| heading right | 0.158 (0.126 to 0.189) | 13.93 (11.0 to 15.4) | 0.240 |
| heading down | 0.129 (0.100 to 0.160) | 14.37 (13.5 to 15.0) | 0.219 |
| heading left | 0.144 (0.100 to 0.186) | 14.34 (12.7 to 15.1) | 0.233 |
| food up | 0.182 (0.125 to 0.224) | 11.29 (3.5 to 15.5) | 0.343 |
| food right | 0.157 (0.101 to 0.212) | 14.98 (13.8 to 16.4) | 0.350 |
| food down | 0.159 (0.126 to 0.203) | 15.42 (14.7 to 16.4) | 0.360 |
| food left | 0.172 (0.122 to 0.222) | 12.98 (7.6 to 15.3) | 0.347 |

  The networks scored 16.02, 15.45, 15.07, 15.38 and 15.37 on the same games without any input frozen.
- **What this says, and how far to trust it:**
  - **Danger ahead is the input the networks cannot do without.** Freezing it (at its average, about "no danger") costs about 4 foods on average: the five networks scored between 10.9 and 11.9 with it frozen, against 15.1 to 16.0 normally. The other two danger inputs cost about 3.6 and 3.2 foods on average.
  - **The food inputs matter very unevenly between networks.** Freezing "food up" left one network at 3.5 foods and another at 15.5 (no loss). The networks do not all use the same inputs.
  - **The heading inputs matter least** when frozen (about 0.5 to 1.5 foods lost), though shuffling them changes 13% to 16% of the moves.
  - **The three measures do not give one ranking.** Shuffling changes 8% to 18% of the moves for every input; freezing and the gradient rank the inputs differently from it.
  - **The five networks disagree with each other.** The rank correlation between networks of the same recipe was 0.51 on average for shuffling (lowest pair 0.31), 0.37 for freezing (lowest 0.07) and 0.46 for the gradient (lowest −0.01). So "importance" is a property of one trained network, not of the recipe.
  - **The networks mostly act like the rule.** They picked the same move as the hand-written rule in 93.6% of the visited states (91.0% to 96.1% by network), and **never** stepped into an immediate danger when a safe move existed. Yet every game still ended in a crash (see [[Snake DQN]]), so the mistakes come from situations the 11 inputs cannot show (the snake's own body further away). I did not look at those games.
- **Not done:** no look at neurons or layers, no integrated gradients, no retrained copies beyond the five, and no check of why the inputs differ between networks.

## What you need
- Python 3.9.6, `torch` 2.8.0 (CPU), `numpy` 1.24.4, `scipy` 1.8.1 (my existing environment on Windows; no fresh virtual environment).
- From the [[Snake DQN]] project, in the same folder: `snake.py`, `dqn.py` and the folder `nets` with the 20 saved networks (this project reads the five `full_DQN_*.pt`).
- A few minutes on a laptop CPU. No GPU, no download.

## Steps
1. **Train a plain table on the same game.** `table_q.py` stores one row of 3 scores for every different 11-number observation, with the same exploration schedule (1.0 to 0.05 over 20,000 steps) and the ordinary Q-learning update $Q\leftarrow Q+0.1\,(r+0.9\max Q'-Q)$. It trains 5 tables at 60,000 and at 300,000 steps and tests each on the 200 games:

```python
# file: table_q.py
import numpy as np
from snake import Snake, play

TEST = range(1000, 1200)


def train_table(seed, steps=60000, eps_steps=20000, eps_end=0.05, gamma=0.9, alpha=0.1):
    rng = np.random.default_rng(seed); env = Snake(); o = env.reset(seed=seed); Q = {}
    for t in range(1, steps + 1):
        q = Q.setdefault(tuple(o), np.zeros(3))
        eps = max(eps_end, 1 - (1 - eps_end) * t / eps_steps)             # the same epsilon schedule as the DQN
        a = int(rng.integers(3)) if rng.random() < eps else int(q.argmax())
        o2, r, done = env.step(a)
        nxt = 0.0 if done else Q.setdefault(tuple(o2), np.zeros(3)).max()
        q[a] += alpha * (r + gamma * nxt - q[a])                          # the ordinary Q-learning update, one table row per observation
        o = env.reset() if done else o2
    return Q


if __name__ == "__main__":
    print("tabular Q-learning on the same 11 numbers, same epsilon schedule, 5 runs each, 200 test games")
    for steps in (60000, 300000):
        means, rows = [], []
        for seed in range(5):
            Q = train_table(seed, steps=steps)
            pol = lambda o: int(Q[tuple(o)].argmax()) if tuple(o) in Q else 0
            means.append(float(np.mean(play(pol, TEST)))); rows.append(len(Q))
        print(f"{steps:>7,} steps: per-run means {' '.join(f'{m:.2f}' for m in means)} | mean {np.mean(means):.2f}, std {np.std(means):.2f} | table rows used {min(rows)} to {max(rows)} of at most 256")
    print("for comparison (Snake DQN project, 60,000 steps): rule 14.85, full DQN 15.46")
```

```text
tabular Q-learning on the same 11 numbers, same epsilon schedule, 5 runs each, 200 test games
 60,000 steps: per-run means 14.74 12.79 13.46 13.49 13.85 | mean 13.67, std 0.64 | table rows used 251 to 255 of at most 256
300,000 steps: per-run means 14.30 16.45 15.96 16.24 16.25 | mean 15.84, std 0.78 | table rows used 256 to 256 of at most 256
for comparison (Snake DQN project, 60,000 steps): rule 14.85, full DQN 15.46
```

2. **Probe the five networks.** `interp.py` loads the five full DQNs, plays the 200 test games with each, and keeps every state it saw. Then for each of the 11 inputs it computes three things. **Shuffle:** mix up that input's column among the saved states and count how often the greedy move changes ([[Permutation Importance]]). **Freeze:** play 200 new games with that input replaced by its average. **Gradient:** the mean size of the slope of the best move's score with respect to that input ([[Saliency Map]]). Then it compares the five networks by rank correlation, and compares each network's moves with the rule's:

```python
# file: interp.py
import numpy as np
import torch
from scipy.stats import spearmanr
from snake import Snake, rule_greedy
from dqn import make_net, greedy

NAMES = ["danger ahead", "danger right", "danger left", "heading up", "heading right", "heading down", "heading left",
         "food up", "food right", "food down", "food left"]
TEST = range(1000, 1200)


def play(policy, collect=False):
    """200 test games; returns the mean score and (if asked) every observation the policy saw"""
    env, scores, seen = Snake(), [], []
    for s in TEST:
        o, done = env.reset(seed=s), False
        while not done:
            seen.append(o); o, _, done = env.step(policy(o))
        scores.append(env.score)
    return float(np.mean(scores)), np.array(seen)


def blinded(net, j, fill):
    """the network, but feature j is replaced by a fixed number before it looks"""
    def act(o):
        o = o.copy(); o[j] = fill
        with torch.no_grad():
            return int(net(torch.from_numpy(o)).argmax())
    return act


rng = np.random.default_rng(0)
table = {k: [] for k in ("shuffle", "blind", "grad")}; agree, unsafe, base = [], [], []
for seed in range(5):
    net = make_net(); net.load_state_dict(torch.load(f"nets/full_DQN_{seed}.pt"))
    score, X = play(greedy(net)); base.append(score)
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        a0 = net(Xt).argmax(1).numpy()
    shuffle, blind, grad = [], [], []
    xg = Xt.clone().requires_grad_(True); net(xg).max(1).values.sum().backward()
    for j in range(11):
        Xs = X.copy(); Xs[:, j] = X[rng.permutation(len(X)), j]                      # permutation importance: mix up one column
        with torch.no_grad():
            shuffle.append(float((net(torch.from_numpy(Xs)).argmax(1).numpy() != a0).mean()))
        blind.append(play(blinded(net, j, float(X[:, j].mean())))[0])               # play whole games with feature j frozen at its average
        grad.append(float(xg.grad[:, j].abs().mean()))                              # how steeply the best Q-value reacts to feature j
    for k, v in zip(table, (shuffle, blind, grad)): table[k].append(v)
    ar = np.array([rule_greedy(o) for o in X])
    agree.append(float((ar == a0).mean())); unsafe.append(float(np.mean([o[a] == 1 and (o[:3] == 0).any() for o, a in zip(X, a0)])))

print("5 full DQNs from the Snake DQN project, 200 test games; the networks' own mean score:", " ".join(f"{b:.2f}" for b in base))
print(f"{'feature':14s} {'action changes when':>20s} {'score with feature':>20s} {'mean |dQ/dx|':>13s}")
print(f"{'':14s} {'shuffled (share)':>20s} {'frozen (mean score)':>20s}")
for j, n in enumerate(NAMES):
    s, b, g = (np.array(table[k])[:, j] for k in table)
    print(f"{n:14s} {s.mean():>12.3f} ({s.min():.3f}-{s.max():.3f}) {b.mean():>8.2f} ({b.min():.1f}-{b.max():.1f}) {g.mean():>10.3f}")
for k in table:
    rho = [spearmanr(table[k][a], table[k][b])[0] for a in range(5) for b in range(a + 1, 5)]
    print(f"agreement between the 5 networks on the feature ranking by '{k}': mean Spearman rho {np.mean(rho):.2f} (min {min(rho):.2f})")
print(f"share of visited states where the network picks the same move as the rule: {np.mean(agree):.3f} (runs {min(agree):.3f} to {max(agree):.3f})")
print(f"share of visited states where the network steps into an immediate danger although a safe move exists: {np.mean(unsafe):.4f}")
```

```text
5 full DQNs from the Snake DQN project, 200 test games; the networks' own mean score: 16.02 15.45 15.07 15.38 15.37
feature         action changes when   score with feature  mean |dQ/dx|
                   shuffled (share)  frozen (mean score)
danger ahead          0.108 (0.100-0.114)    11.41 (10.9-11.9)      0.423
danger right          0.083 (0.070-0.098)    11.86 (10.7-12.5)      0.310
danger left           0.094 (0.065-0.115)    12.22 (10.7-13.4)      0.324
heading up            0.163 (0.115-0.192)    14.93 (13.2-16.0)      0.240
heading right         0.158 (0.126-0.189)    13.93 (11.0-15.4)      0.240
heading down          0.129 (0.100-0.160)    14.37 (13.5-15.0)      0.219
heading left          0.144 (0.100-0.186)    14.34 (12.7-15.1)      0.233
food up               0.182 (0.125-0.224)    11.29 (3.5-15.5)      0.343
food right            0.157 (0.101-0.212)    14.98 (13.8-16.4)      0.350
food down             0.159 (0.126-0.203)    15.42 (14.7-16.4)      0.360
food left             0.172 (0.122-0.222)    12.98 (7.6-15.3)      0.347
agreement between the 5 networks on the feature ranking by 'shuffle': mean Spearman rho 0.51 (min 0.31)
agreement between the 5 networks on the feature ranking by 'blind': mean Spearman rho 0.37 (min 0.07)
agreement between the 5 networks on the feature ranking by 'grad': mean Spearman rho 0.46 (min -0.01)
share of visited states where the network picks the same move as the rule: 0.936 (runs 0.910 to 0.961)
share of visited states where the network steps into an immediate danger although a safe move exists: 0.0000
```

3. **Read it** with the cautions below. The first line shows that the networks are the same ones as in [[Snake DQN]] (the same five scores), and the table shows the three measures side by side.

## What went wrong, and the fix
- **A 60,000-step table looked like proof that the network was better.** It scored 13.67 against 15.46. Before stating that, I ran the table for 300,000 steps and it reached 15.84. The honest statement is "faster, not better". I did that second run after seeing the first result, so it is not a clean pre-planned test.
- **Freezing a one-hot input is not clean.** The four heading inputs are a one-hot group (exactly one is 1). Freezing one of them at its average (about 0.25) gives the network a value it never saw in play, and shuffling one creates impossible headings (see [[Interpretability]]: 36.6% of rows for "heading up"). So the heading rows of the table are the least trustworthy, and the small score loss for freezing them could be partly an artefact.
- **The measures disagree.** That is not an error to fix; it is a finding, and the note says so.
- **Nothing else broke.** The five scores printed on the first line of `interp.py` match the five scores in [[Snake DQN]].

## What the measures cannot tell
- **Cause.** They show what a trained network uses or is sensitive to, not why it learned to.
- **A rule it follows.** 93.6% agreement with the rule does not mean the network contains the rule; it only means they pick the same move most of the time.
- **Anything about the 6.4% of moves where they differ**, which is where the network's small edge, if real, must come from. I did not look at them.

## What I did not run
- **Neurons, layers or circuits;** integrated gradients, SmoothGrad or other attribution methods; SHAP or LIME.
- **More than five trained networks,** or retrained copies with other settings, so I cannot say how much the importance rankings vary across training.
- **A tuned table** (learning rate, exploration), or a table with the same number of updates as the network (the network updates 64 samples per step; the table one).
- **Error bars** on the rank correlations (five networks give only ten pairs) and on the score tables, which are means over networks with the lowest and highest in brackets.
- **The games that end in a crash:** I did not examine why they happen.
- **A bigger board or an image-based network,** where the inputs are not so few.

## Re-run check
Both scripts were extracted from this note into a fresh, empty folder (with copies of `snake.py`, `dqn.py` and the saved networks from [[Snake DQN]]) and run again. The output of `table_q.py` (all ten table scores) and of `interp.py` (every number in the table, the rank correlations and the agreement with the rule) matched the blocks above exactly, so the results are repeatable on this machine. The snippets in [[Interpretability]], [[Permutation Importance]] and [[Saliency Map]] were re-run the same way and matched. Not tested: another machine or other library versions.

## Concepts used
- [[Interpretability]] · [[Permutation Importance]] · [[Saliency Map]] · [[Deep Q-Network]] · [[Q-Learning]] · [[Neural Network]] · [[Backpropagation]] · [[Snake DQN]] · [[Epsilon-Greedy]]

## Next ideas
- Retrain the networks with 20 seeds and measure how much the importance rankings really vary (and put error bars on them with [[Bootstrap]]).
- Look at the 6.4% of states where the network and the rule disagree: which of them does better?
- Look at the games that end in a crash and find what the network cannot see.
- Try integrated gradients and a one-hot-safe way of removing an input group.

## Sources
- My own code and runs (versions above). The tools are standard ideas written from memory and checked on small toys in the concept notes, not against a textbook.
- No images in this note.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-rl What was the setup of Looking Inside Snake DQN?
?
Five trained Snake DQNs were probed with three tools per input (shuffle the input and count changed moves; freeze it at its average and play 200 games; mean gradient size), compared across networks by rank correlation and with the hand-written rule; a plain Q-table with the same exploration was trained on the same 11 inputs to see whether a network was needed.

#flashcards/projects/concept #flashcards/topic/project-rl Was a neural network needed for Snake with 11 inputs?
?
Not for the final level. A table scored 13.67 after 60,000 steps against 15.46 for the DQN, and 15.84 after 300,000 steps (the gap to the DQN, 0.4, is below the spread between runs of 0.8). The network learned faster. The 300,000-step run was added after seeing the first result and the table was not tuned.

#flashcards/projects/concept #flashcards/topic/project-rl Which input could the Snake DQNs not do without?
?
"Danger ahead": freezing it at its average cost about 4 foods (the five networks scored 10.9 to 11.9 with it frozen, against 15.1 to 16.0 normally). The other danger inputs cost about 3.2 to 3.6. The food inputs mattered very unevenly between networks (freezing "food up" left one network at 3.5 foods and another at 15.5).

#flashcards/projects/concept #flashcards/topic/project-rl Did the three measures and the five networks agree on which inputs matter?
?
Only moderately. Rank correlation between networks averaged 0.51 for shuffling (lowest 0.31), 0.37 for freezing (lowest 0.07) and 0.46 for the gradient (lowest −0.01). Importance belongs to one trained network, not to the recipe.

#flashcards/projects/concept #flashcards/topic/project-rl How much did the Snake DQN act like the hand-written rule?
?
It chose the same move in 93.6% of visited states (91.0% to 96.1% by network) and never stepped into an immediate danger when a safe move existed. Its crashes come from situations the 11 inputs cannot show. I did not examine those games or the 6.4% of states where it differs.

#flashcards/projects/concept #flashcards/topic/project-rl Why are the heading rows of the importance table the least trustworthy?
?
The four heading inputs are a one-hot group. Freezing one at its average (about 0.25) is a value never seen in play, and shuffling one creates impossible headings (36.6% of rows for "heading up"), so the small effects could be partly an artefact.

#flashcards/projects/concept #flashcards/topic/project-rl What did Looking Inside Snake DQN not run?
?
Neurons or layers; integrated gradients, SHAP or LIME; more than five networks; a tuned table or one with equal updates; error bars on the rank correlations; the crash games; a bigger board or an image network.

#flashcards/projects/cloze #flashcards/topic/project-rl In Looking Inside Snake DQN the plain table was ==slower== to learn than the network but reached about the same score after ==300,000== steps.
