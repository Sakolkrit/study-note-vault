---
tags: [project, reinforcement-learning, deep-learning, dqn, pytorch, snake, game]
status: done
stack: Python 3.9.6, PyTorch 2.8.0 (CPU only), NumPy 1.24.4, matplotlib 3.7.1, Pillow 10.4.0
goal: teach a small neural network to play Snake with deep Q-learning on an ordinary laptop CPU, compare it with simple players, and test what the replay buffer and the target network change
---

# Snake DQN

> **In one line:** write the game Snake as a tiny environment, train a [[Deep Q-Network]] to play it on a laptop **without a GPU**, and measure it against a random player, a "safe random" player and a hand-written rule, with and without its two helpers ([[Experience Replay]] and the target network).

![[nokia-3310-snake-ii.jpg]]
*Snake II on a Nokia 3310, the game this project copies in a tiny version. Photo by Toteemipaalu, CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Matopeli.jpg). I shrank the file to 900 × 1200 pixels. The photo is of the phone game, not of my code.*

## Goal
Done means: (1) a game written as an environment with `reset` and `step`, (2) a DQN trained from scratch in PyTorch on the CPU, (3) a fair comparison on 200 test games that are the same for every player, (4) four versions of the DQN (with and without the target network, with and without replay) on 5 training seeds each, (5) a learning-curve picture, a GIF of the trained snake, and a check of whether the gaps are bigger than the noise, and (6) a plain list of what was not run. The vault's [[Q-Learning Grid World]] used a table; this project replaces the table with a [[Neural Network]].

## Result in short
- **The game:** an 8 × 8 board, a snake of length 3 at the start, one piece of food. Three moves: straight, turn right, turn left. Reward +1 for food, −1 for dying (wall or own body) or for 100 steps without food, 0 otherwise. The score of a game is the number of foods eaten.
- **What the network sees:** only 11 numbers: is there danger straight ahead, to the right, to the left (3), which way the snake is heading (4), and where the food is (up, right, down, left; 4). It does **not** see the board. This makes the problem small enough for a laptop, and it also makes it easy.
- **Players on 200 test games** (mean foods eaten, higher is better):

| Player | Mean score |
|---|---|
| random moves | 0.22 |
| random safe moves (never steps into an immediate danger) | 2.17 |
| rule: among safe moves, turn toward the food | 14.85 |
| **full DQN** (5 runs) | **15.46** (std 0.31 between runs) |
| DQN without target network | 15.69 (0.51) |
| DQN without replay | 15.52 (0.67) |
| DQN with neither | 15.34 (0.37) |

- **The DQN learned the game:** from 0.22 for random play to about 15.5, and it learned quickly (the full DQN averaged 15.1 after only 5,000 training steps on 30 test games).
- **But it only just beats the hand-written rule.** Each of the five full DQNs scored higher than the rule on the same 200 games, by +0.23 to +1.17, with a standard error of about 0.45 each. That is between half and 2.5 standard errors, and the five networks share the same 200 games, so they are not five independent tests. I call it a small edge that is **not firmly established**. (Tested afterwards in [[Error Bars Lab]]: the five-network average is +0.61 with a 95% interval of +0.02 to +1.18 and $p=0.046$, right on the border.)
- **The four versions end at the same level** (15.3 to 15.7 with a spread of 0.3 to 0.7 between seeds), so at 60,000 steps I see no difference between them.
- **They differ in speed.** After 5,000 steps (30 test games, mean of 5 runs): full DQN 15.1, no target network 14.7, no replay 4.9, neither 9.4. The replay buffer mattered for early learning; the target network did not show an effect here (this does not mean it is useless in harder games).
- **Cost on a laptop with no GPU:** about 485 seconds per training run (6 runs at the same time on 8 logical cores), and about 32 minutes for all 20 runs. A single run alone took about 52 seconds for 20,000 steps in a quick check.
- **A caution about the whole project:** the 11 numbers can take only a few hundred different values (at most 256 by counting; 164 turned up in 200,000 random steps, see step 8). A plain table of scores, with one row per value, would be small enough for ordinary [[Q-Learning]]. I did **not** run that table here, so this project does not show that a neural network was needed, only that one works. (Run afterwards in [[Looking Inside Snake DQN]]: the table needed more steps, scoring 13.67 at 60,000 steps and 15.84 at 300,000, against 15.46 for the network.)
- **Not done:** only one board size and one set of settings, no tuning, no image-based network, no Atari, no GPU, no table version.

![[snake-dqn-learning-curves.png]]
*Learning curves from my run: foods eaten on 30 test games every 5,000 steps, mean of 5 runs per version, shaded band = lowest to highest run. The dashed line is the rule player on the same 30 games (16.4), which is higher than its 14.85 on the 200 test games, a hint of how much 30-game averages jump around. Made with `plot.py` below.*

![[snake-dqn-agent.gif]]
*One game of the trained snake (full DQN, seed 0, game seed 1000, score 17). The red dot is the food and the dark green square is the head. Made with `play.py` below.*

## What you need
- Python 3.9.6, `torch` 2.8.0 (the CPU build; `torch.cuda.is_available()` was `False`), `numpy` 1.24.4, `matplotlib` 3.7.1, `Pillow` 10.4.0 (my existing Python environment on Windows; no fresh virtual environment).
- A laptop with 8 logical cores and no GPU. No data and no downloads: the game makes its own data.
- About 35 minutes of free CPU for `run_all.py`. For a quick try, lower `steps` in `dqn.py` to 20,000.

## Steps
Put all files in one folder.

1. **Write the game** as an environment ([[Markov Decision Process]]). `reset()` starts a game and returns the 11 numbers, `step(action)` returns the next 11 numbers, the reward and whether the game is over. The file also has `play()` (one game per seed) and the rule player `rule_greedy`, which only uses the same 11 numbers as the network.

```python
# file: snake.py
import random
import numpy as np

N = 8                                              # board is N x N
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left as (row, column)
TURN = [0, 1, 3]                                   # action 0 = straight, 1 = turn right, 2 = turn left


class Snake:
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        m = N // 2
        self.body = [(m, m), (m, m - 1), (m, m - 2)]   # head first, heading right
        self.d, self.score, self.since = 1, 0, 0
        self.food = self.rng.choice([(r, c) for r in range(N) for c in range(N) if (r, c) not in self.body])
        return self.obs()

    def danger(self, cell):                        # wall, or body (the tail cell is free because it moves away)
        return not (0 <= cell[0] < N and 0 <= cell[1] < N) or cell in self.body[:-1]

    def obs(self):
        (r, c), d = self.body[0], self.d
        ahead = [(r + DIRS[(d + t) % 4][0], c + DIRS[(d + t) % 4][1]) for t in TURN]
        heading = [int(d == k) for k in range(4)]
        fr, fc = self.food
        return np.array([self.danger(x) for x in ahead] + heading + [fr < r, fc > c, fr > r, fc < c], dtype=np.float32)

    def step(self, action):
        """reward +1 for food, -1 for dying or for 100 steps without food, 0 otherwise"""
        self.d = (self.d + TURN[action]) % 4
        head = (self.body[0][0] + DIRS[self.d][0], self.body[0][1] + DIRS[self.d][1])
        if self.danger(head):
            return self.obs(), -1.0, True
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1; self.since = 0
            free = [(r, c) for r in range(N) for c in range(N) if (r, c) not in self.body]
            if not free:                           # the board is full: the game is won
                return self.obs(), 1.0, True
            self.food = self.rng.choice(free)
            return self.obs(), 1.0, False
        self.body.pop(); self.since += 1
        return self.obs(), (-1.0 if self.since >= 100 else 0.0), self.since >= 100


def play(policy, seeds):
    """play one game per seed with the given policy (observation -> action); returns the scores"""
    env, scores = Snake(), []
    for s in seeds:
        o, done = env.reset(seed=s), False
        while not done:
            o, _, done = env.step(policy(o))
        scores.append(env.score)
    return scores


def rule_greedy(o):
    """no learning: among the safe moves take one that turns towards the food, else any safe move, else straight"""
    d = int(np.argmax(o[3:7]))
    safe = [a for a in range(3) if o[a] == 0]
    toward = [a for a in safe if o[7 + (d + TURN[a]) % 4] == 1]
    return (toward or safe or [0])[0]
```

2. **Write the DQN** ([[Deep Q-Network]]): a network with two hidden layers of 64 units ([[ReLU]]) that outputs one score per move; a replay buffer kept in plain arrays ([[Experience Replay]]); a copy of the network that is refreshed every 500 steps to make the targets (the target network); and exploration that falls from 1.0 to 0.05 over 20,000 steps ([[Epsilon-Greedy]]). The loss is the mean squared error between the target and the network's guess, optimised with [[Adam Optimizer]]. `train(seed, target=False)` switches the target network off and `train(seed, replay=False)` makes each update use only the newest step.

```python
# file: dqn.py
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from snake import Snake, play

torch.set_num_threads(1)


def make_net():
    return nn.Sequential(nn.Linear(11, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))


def greedy(net):
    def act(o):
        with torch.no_grad():
            return int(net(torch.from_numpy(o)).argmax())
    return act


def train(seed, target=True, replay=True, steps=60000, eps_steps=20000, eps_end=0.05, gamma=0.9, lr=1e-3,
          batch=64, cap=20000, start=1000, sync=500, check=5000):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    env = Snake(); o = env.reset(seed=seed)
    net, tgt = make_net(), make_net(); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    O, O2 = np.zeros((cap, 11), np.float32), np.zeros((cap, 11), np.float32)       # the replay buffer
    A, R, D = np.zeros(cap, np.int64), np.zeros(cap, np.float32), np.zeros(cap, np.float32)
    act, curve = greedy(net), []
    for t in range(1, steps + 1):
        eps = max(eps_end, 1 - (1 - eps_end) * t / eps_steps)                      # epsilon falls from 1 to 0.05
        a = int(rng.integers(3)) if rng.random() < eps else act(o)
        o2, r, done = env.step(a)
        i = (t - 1) % cap                                                          # overwrite the oldest slot when full
        O[i], A[i], R[i], O2[i], D[i] = o, a, r, o2, done
        o = env.reset() if done else o2
        if t >= start:
            idx = rng.integers(min(t, cap), size=batch) if replay else np.array([i])   # no replay: only the newest step
            q = net(torch.from_numpy(O[idx])).gather(1, torch.from_numpy(A[idx])[:, None])[:, 0]
            with torch.no_grad():
                nxt = (tgt if target else net)(torch.from_numpy(O2[idx])).max(1).values
                y = torch.from_numpy(R[idx]) + gamma * nxt * (1 - torch.from_numpy(D[idx]))
            loss = F.mse_loss(q, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10); opt.step()
            if target and t % sync == 0:
                tgt.load_state_dict(net.state_dict())
        if t % check == 0:
            curve.append((t, float(np.mean(play(act, range(5000, 5030))))))         # 30 greedy test games
    return net, curve
```

3. **Train all four versions on 5 seeds and test every player on the same 200 games** (seeds 1000 to 1199). The training runs are spread over 6 processes. Run `python run_all.py`; it takes about 32 minutes on my laptop and prints one table at the end. The random players are run on 5 seeds, the rule player once (it has no randomness).

```python
# file: run_all.py
import json, os, random, time
import numpy as np
import torch
from multiprocessing import Pool
from snake import play, rule_greedy
from dqn import train, greedy

VARIANTS = {"full DQN": (True, True), "no target network": (False, True), "no replay": (True, False), "neither": (False, False)}
SEEDS, TEST = range(5), range(1000, 1200)           # 5 training seeds; 200 test games, the same for every method


def job(args):
    name, seed = args
    target, replay = VARIANTS[name]
    t0 = time.time()
    net, curve = train(seed, target=target, replay=replay)
    os.makedirs("nets", exist_ok=True)
    torch.save(net.state_dict(), f"nets/{name.replace(' ', '_')}_{seed}.pt")
    return name, seed, float(np.mean(play(greedy(net), TEST))), curve, time.time() - t0


if __name__ == "__main__":
    with Pool(6) as p:
        res = p.map(job, [(n, s) for n in VARIANTS for s in SEEDS])
    json.dump([[n, s, m, c, t] for n, s, m, c, t in res], open("results.json", "w"))
    print("Snake 8x8; mean foods eaten per game; 200 test games (seeds 1000-1199), the same games for every method")
    print(f"{'method':<20}{'mean over 5 runs':>18}{'std':>8}   per-run means")
    rows = {}
    for rep_name, mk in [("random moves", lambda r: (lambda o: r.randrange(3))),
                         ("random safe moves", lambda r: (lambda o: r.choice([a for a in range(3) if o[a] == 0] or [0])))]:
        rows[rep_name] = [float(np.mean(play(mk(random.Random(k)), TEST))) for k in range(5)]
    rows["rule: toward food"] = [float(np.mean(play(rule_greedy, TEST)))]
    for n in VARIANTS:
        rows[n] = [m for nn, s, m, c, t in res if nn == n]
    for n, v in rows.items():
        print(f"{n:<20}{np.mean(v):>18.2f}{(np.std(v) if len(v) > 1 else 0):>8.2f}   " + " ".join(f"{x:.2f}" for x in v))
    print(f"mean training time per DQN run: {np.mean([t for *_, t in res]):.0f} s (6 runs at once on 8 cores)")
```

```text
Snake 8x8; mean foods eaten per game; 200 test games (seeds 1000-1199), the same games for every method
method                mean over 5 runs     std   per-run means
random moves                      0.22    0.02   0.21 0.19 0.23 0.21 0.23
random safe moves                 2.17    0.11   2.06 2.12 2.09 2.22 2.35
rule: toward food                14.85    0.00   14.85
full DQN                         15.46    0.31   16.02 15.45 15.07 15.38 15.37
no target network                15.69    0.51   16.59 15.17 15.87 15.47 15.35
no replay                        15.52    0.67   15.79 14.43 15.21 16.42 15.78
neither                          15.34    0.37   15.66 15.48 15.23 14.67 15.66
mean training time per DQN run: 485 s (6 runs at once on 8 cores)
```

4. **Read it.** The three lines at the top set the scale: random play scores 0.22, a player that only avoids walls and its own body scores 2.17, and the simple rule scores 14.85, so avoiding danger and heading to the food is most of the game. The DQN rows are all close to 15.5. The differences between the four DQN versions (15.34 to 15.69) are smaller than the differences between seeds of one version.

5. **Draw the learning curves** (`python plot.py`, which also prints the rule player's score on the 30 checkpoint games):

```python
# file: plot.py
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from snake import play, rule_greedy

res = json.load(open("results.json"))
base = np.mean(play(rule_greedy, range(5000, 5030)))             # the rule player on the same 30 checkpoint games
fig, ax = plt.subplots(figsize=(7, 4))
for name in dict.fromkeys(r[0] for r in res):
    runs = np.array([[v for _, v in r[3]] for r in res if r[0] == name]); x = [t for t, _ in res[0][3]]
    ax.plot(x, runs.mean(0), label=name); ax.fill_between(x, runs.min(0), runs.max(0), alpha=0.15)
ax.axhline(base, color="k", ls="--", label=f"rule: toward food ({base:.1f})")
ax.set_xlabel("training steps"); ax.set_ylabel("foods eaten per game (30 test games)")
ax.set_title("Snake 8x8: mean of 5 runs, band = min to max"); ax.legend(); fig.tight_layout(); fig.savefig("curves.png", dpi=130)
print("saved curves.png; rule player on the 30 checkpoint games:", round(float(base), 2))
```

```text
saved curves.png; rule player on the 30 checkpoint games: 16.43
```

6. **Watch it play.** `play.py` loads the full DQN of seed 0, plays game 1000 and saves the GIF above:

```python
# file: play.py
import torch
from PIL import Image, ImageDraw
from snake import Snake, N
from dqn import make_net, greedy

net = make_net(); net.load_state_dict(torch.load("nets/full_DQN_0.pt")); act = greedy(net)
env, C, frames = Snake(), 24, []
o, done = env.reset(seed=1000), False
while not done:
    im = Image.new("RGB", (N * C, N * C), "white"); d = ImageDraw.Draw(im)
    for k, (r, c) in enumerate(env.body):
        d.rectangle([c * C + 1, r * C + 1, c * C + C - 2, r * C + C - 2], fill="#1a7f37" if k else "#0b3d1a")
    r, c = env.food; d.ellipse([c * C + 4, r * C + 4, c * C + C - 5, r * C + C - 5], fill="#d1242f")
    frames.append(im); o, _, done = env.step(act(o))
frames[0].save("agent.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
print("game seed 1000: score", env.score, "| frames", len(frames))
```

```text
game seed 1000: score 17 | frames 100
```

7. **Ask whether the gap to the rule is real.** `stats.py` plays the 200 test games again with the rule and with each of the five full DQNs, and compares them game by game on the same games (a paired difference):

```python
# file: stats.py
import glob
import numpy as np
import torch
from snake import Snake, rule_greedy
from dqn import make_net, greedy

TEST = range(1000, 1200)


def games(policy):
    """per-game score, and whether the game ended by the 100-steps-without-food rule"""
    env, out = Snake(), []
    for s in TEST:
        o, done = env.reset(seed=s), False
        while not done:
            o, _, done = env.step(policy(o))
        out.append((env.score, env.since >= 100))
    return np.array(out, dtype=float)


rule = games(rule_greedy)
print(f"rule player: mean {rule[:, 0].mean():.2f}, std of one game {rule[:, 0].std(ddof=1):.2f}, "
      f"standard error over 200 games {rule[:, 0].std(ddof=1) / 200 ** 0.5:.2f}, games ended by time-out {int(rule[:, 1].sum())}")
for f in sorted(glob.glob("nets/full_DQN_*.pt")):
    net = make_net(); net.load_state_dict(torch.load(f))
    g = games(greedy(net)); d = g[:, 0] - rule[:, 0]                # paired: the same 200 games for both players
    print(f"{f[5:-3]:<11} mean {g[:, 0].mean():.2f} | minus rule {d.mean():+.2f} (standard error {d.std(ddof=1) / 200 ** 0.5:.2f}) | time-outs {int(g[:, 1].sum())}")
```

```text
rule player: mean 14.85, std of one game 4.69, standard error over 200 games 0.33, games ended by time-out 0
full_DQN_0  mean 16.02 | minus rule +1.17 (standard error 0.46) | time-outs 0
full_DQN_1  mean 15.45 | minus rule +0.59 (standard error 0.47) | time-outs 0
full_DQN_2  mean 15.07 | minus rule +0.23 (standard error 0.42) | time-outs 0
full_DQN_3  mean 15.38 | minus rule +0.53 (standard error 0.45) | time-outs 0
full_DQN_4  mean 15.37 | minus rule +0.52 (standard error 0.42) | time-outs 0
```

   One game's score has a spread of 4.69 for the rule player, so a mean over 200 games has a standard error of 0.33. The five DQNs beat the rule by +0.23 to +1.17 (standard errors around 0.45): the sign is the same each time, but the size is inside what luck in 200 games could give for some of them. No game in the test ended by the 100-step time-out: every game ended by hitting a wall or the snake's own body.

8. **Count how many different things the network can see.** `count_states.py` plays 200,000 random steps and writes down every different 11-number observation:

```python
# file: count_states.py
import random
from snake import Snake

env, rng, seen = Snake(), random.Random(0), set()
o = env.reset(seed=0)
for _ in range(200000):                          # random play; write down every different 11-number observation
    seen.add(tuple(o)); o, _, done = env.step(rng.randrange(3))
    if done:
        o = env.reset()
print("most possible, by counting:", 8 * 4 * 8, "(3 danger flags, 4 headings, 8 food directions)")
print("different observations seen in 200,000 random steps:", len(seen))
```

```text
most possible, by counting: 256 (3 danger flags, 4 headings, 8 food directions)
different observations seen in 200,000 random steps: 164
```

   So the "state space" here is tiny. In Snake on a bigger board, or with the whole board as the input, it would not be, and that is where the network earns its place.

## What went wrong, and the fix
- **The first version of the idea was larger.** I planned a network that sees the whole board. I chose the 11-number description instead, because a board-image network would need much more training than a laptop CPU can give in a short run. The price is that the task is easy and a hand-written rule is already strong.
- **My first estimate of the run time was too short.** One run alone took about 52 seconds for 20,000 steps, but with 6 runs at once each took about 485 seconds for 60,000 steps (the cores and memory are shared), so the whole job took 32 minutes, not the 6 to 10 I first guessed.
- **A background-job notice said "completed" while the training was still running.** It was the shell that had started the job, not the job. I checked the process list and the output file before trusting it.
- **One image was rejected.** A Commons picture found by searching for "Deep Q Learning" turned out to be an adult photo, not a diagram. I did not use it.
- **Nothing else broke.** The script ran the first time and the numbers from `stats.py` (16.02, 15.45, 15.07, 15.38, 15.37 for the five full DQNs) match the first numbers printed by `run_all.py`, which is a check that the saved networks are the ones that were tested.

## What I did not run
- **A table (tabular Q-learning) on the same 11 numbers.** It would likely do about as well, but I did not run it.
- **Any harder version:** a bigger board, a network that sees the board as an image (a [[Convolutional Neural Network]]), or Atari games.
- **Variants of DQN:** double DQN, dueling networks, prioritised replay, a soft target update, the Huber loss (I used the squared error).
- **Tuning** of the learning rate, the discount (0.9), the buffer size (20,000), the batch (64), the sync period (500) or the exploration schedule.
- **More than 5 seeds**, or any significance test beyond the standard errors above.
- **Any other reward** (for example a small step cost or a bonus for moving toward food).
- **A GPU run.** The aim was to see what a laptop with no GPU can do.
- **Why** the replay buffer helped early learning and the target network showed no effect. My guess is that the replay buffer lets each step be used about 63 times instead of once (my arithmetic, see [[Experience Replay]]), but I did not test that.
- **The 100-step time-out in training.** I treat it as a game over with reward −1. No test game hit it, but I did not count how often it happened during training.

## Re-run check
Every code block in this note was extracted into a fresh, empty folder and run again in order, a second time from scratch (including all 20 training runs, about 32 minutes). The output of `run_all.py` matched the output above in every number, including each per-run score; only the seconds figure differed and it is not compared. The outputs of `plot.py`, `play.py`, `stats.py` and `count_states.py` matched too. Training is repeatable with these seeds on this machine (one thread per process); I did not test another machine or another PyTorch version, where small differences in the arithmetic could change the scores. The snippets in [[Deep Q-Network]], [[Experience Replay]] and [[Epsilon-Greedy]] were re-run the same way and matched.

## Concepts used
- [[Deep Q-Network]] · [[Experience Replay]] · [[Epsilon-Greedy]] · [[Q-Learning]] · [[Reinforcement Learning]] · [[Markov Decision Process]] · [[Neural Network]] · [[PyTorch]] · [[Adam Optimizer]] · [[ReLU]] · [[Loss Function]] · [[Learning Rate]] · [[Overfitting]] · [[Q-Learning Grid World]]
- [[Error Bars Lab]] · [[Hypothesis Testing]]
- [[Looking Inside Snake DQN]] · [[Interpretability]]
## Next ideas
- Run plain tabular [[Q-Learning]] on the same 11 numbers and compare it with the DQN.
- Make the board bigger (12 × 12 or 16 × 16), where the body gets in the way and the 11 numbers are not enough, and see whether the rule player falls behind the DQN.
- Add a board-image network and see how many steps it needs.
- Try double DQN and the Huber loss.
- Give each version 20 seeds so that the small gaps can be tested.
- A [[Policy Gradient]] or [[PPO]] player on the same game.

## Sources
- My own code and runs (versions above). The Nokia photo: Toteemipaalu, CC BY-SA 4.0, [File:Matopeli.jpg](https://commons.wikimedia.org/wiki/File:Matopeli.jpg) on Wikimedia Commons.
- The DQN idea and the 11-number description of Snake are from memory; I did not read the original papers for this project and I did not copy code from anywhere.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-rl What was the setup of the Snake DQN project?
?
An 8 × 8 Snake game written as an environment. The network sees 11 numbers (3 danger flags, 4 heading, 4 food direction), picks straight, right or left, and gets +1 for food and −1 for dying or 100 steps without food. A small network was trained with DQN on the CPU, 4 versions × 5 seeds, and tested on the same 200 games.

#flashcards/projects/concept #flashcards/topic/project-rl What were the scores of the random, safe-random and rule players on the 200 test games?
?
Random moves 0.22, random safe moves 2.17, and the rule (safe moves, turn toward the food) 14.85 foods per game.

#flashcards/projects/concept #flashcards/topic/project-rl What was the main finding of the Snake DQN project?
?
The full DQN reached 15.46 foods per game (5 runs, std 0.31) against 14.85 for the hand-written rule. The edge was +0.23 to +1.17 per network with a standard error of about 0.45 each, so it is small and not firmly established. The easy 11-number view of the game makes a simple rule strong.

#flashcards/projects/concept #flashcards/topic/project-rl What did removing the replay buffer or the target network change in Snake DQN?
?
At 60,000 steps nothing visible: all four versions scored 15.3 to 15.7. Early on it mattered: after 5,000 steps full DQN 15.1, no target network 14.7, no replay 4.9, neither 9.4 (30 test games, 5 runs). So replay sped up learning; the target network showed no effect here.

#flashcards/projects/concept #flashcards/topic/project-rl How long did training take on a laptop with no GPU, and what did that depend on?
?
About 485 seconds per run when 6 runs shared 8 logical cores, and about 32 minutes for all 20 runs. One run alone took about 52 seconds for 20,000 steps. The tiny network and the 11-number input are what make this possible on a CPU.

#flashcards/projects/concept #flashcards/topic/project-rl Why is the Snake DQN result a weak test of deep Q-learning?
?
The network sees only 11 hand-made numbers, not the board, so the game is easy and a hand-written rule already scores 14.85. One board size, no tuning, 5 seeds, and the five networks share the same 200 test games.

#flashcards/projects/concept #flashcards/topic/project-rl Why could a plain Q-table have been used in Snake DQN, and was it run?
?
The 11 numbers take at most 8 × 4 × 8 = 256 different values (my arithmetic; 164 appeared in 200,000 random steps), so a table with one row per value is small. It was not run, so the project does not show that a neural network was needed.

#flashcards/projects/concept #flashcards/topic/project-rl Name things the Snake DQN project did not run.
?
A table version of Q-learning, a bigger board, an image-based network, Atari, double or dueling DQN, prioritised replay, the Huber loss, any tuning, more than 5 seeds, other rewards, a GPU run, and tests of why replay helped early learning.

#flashcards/projects/cloze #flashcards/topic/project-rl In Snake DQN all four versions ended at the same level, but without ==replay== the early learning was much ==slower==.

#flashcards/projects/image #flashcards/topic/project-rl ![[snake-dqn-learning-curves.png]]
What does the learning-curve picture show?
?
Foods eaten on 30 test games every 5,000 training steps for the four DQN versions (mean of 5 runs, band = lowest to highest). All reach about the level of the rule player (dashed, 16.4 on those 30 games) by 20,000 steps; the versions without replay start much lower.

#flashcards/projects/image #flashcards/topic/project-rl ![[snake-dqn-agent.gif]]
What does the GIF show?
?
One game of the trained snake (full DQN, seed 0, game seed 1000, score 17): red dot = food, dark green = head. It is my own output, not an outside picture.

#flashcards/projects/image #flashcards/topic/project-rl ![[nokia-3310-snake-ii.jpg]]
What is in this photo and why is it in the note?
?
A Nokia 3310 playing Snake II (photo by Toteemipaalu, CC BY-SA 4.0, Wikimedia Commons). It shows the original game this project copies in a tiny version; it is not from my code.
