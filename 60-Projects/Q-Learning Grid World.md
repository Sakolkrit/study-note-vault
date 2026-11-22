---
tags: [project, python, numpy, reinforcement-learning, q-learning, value-iteration, mdp]
status: done
stack: Python 3.9, NumPy
goal: compute the exact best answer for a small grid world with value iteration, then let Q-learning learn it by trial and error and measure how far it gets, and what changes that
---

# Q-Learning Grid World

> **In one line:** solve a small cliff-walking world two ways, exactly (by [[Dynamic Programming]] on the [[Bellman Equation]]) and by trial and error ([[Q-Learning]]), then measure the gap between the learned answer and the exact one.

## Goal
Done means five checks pass: (1) the world is written as a table of transitions, as in a [[Markov Decision Process]], (2) value iteration finds the best value of every square, and that answer is checked three ways, (3) Q-learning reaches the exact answer in the world with no randomness, (4) in a slippery version, eight settings are compared on 5 seeds each against the exact answer, and (5) what is and is not shown is written down. The vault's [[Q-Learning]] note already shows a cliff-walking run with SARSA. This project adds the exact answer to measure against, a slippery version, and the settings comparison.

## Result in short
- **The world:** a grid of 4 rows and 12 columns. You start bottom-left and the goal is bottom-right. Each step costs 1. The 10 squares between them on the bottom row are a cliff: stepping on one costs 100 and sends you back to the start. That leaves 37 squares to stand on (not counting the goal). Four moves: up, right, down, left. In the slippery version, with chance 0.2 your move is replaced by a random one of the four.
- **Exact answer (value iteration):** with no slipping, the best total score from the start is −13.000 after 15 sweeps, going along the row beside the cliff. With slipping 0.2 it is −28.634 after 119 sweeps, and the best route switches to the top row, away from the cliff.
- **That exact answer was checked:** solving the best policy's equations directly differs by 3.2e-10, the equations hold to 6.5e-11, and 20,000 simulated episodes of that policy averaged −28.417 with a standard error of 0.204 (about one standard error from −28.634).
- **Q-learning with no slipping** found the exact optimum on all 5 seeds: greedy-policy score −13.00 each time.
- **With slipping 0.2** (1,000 episodes, learning rate 0.5, 10% random moves): the greedy policy scored −42.67 on average against the exact −28.63. Four seeds were between −28.3 and −35.0, and one seed scored −90.5.
- **What helped most:** a small learning rate. A learning rate of 0.05 gave −29.67 (all five seeds between −28.6 and −32.2), and a learning rate of 1.0 gave −129.39 (all five seeds worse than −74). 5,000 episodes also helped (−29.47).
- **Not clear from 5 seeds:** the effect of the random-move rate (10% random moves: −42.67; 1%: −120.87; 30%: −109.58). The means for 1% and 30% each come from a single failed seed (−427.8 and −410.1), so I cannot rank them.
- **Not done:** no neural-network version here (one was built later, on the game Snake: [[Snake DQN]]), no SARSA here (see [[Q-Learning]]), no decaying learning rate, only 5 seeds, and only one grid.

## What you need
Versions I used: my existing Python environment on Windows, no fresh virtual environment (only NumPy is needed, but I did not make a new one).

| Tool | Version |
|---|---|
| Python | 3.9.6 (the code uses the `:=` operator, which needs 3.8 or later) |
| numpy | 1.24.4 |

No data and no downloads. Everything is made by the code.

## Steps
All files go in one folder.

1. **Write the world as a table of what can happen** ([[Markov Decision Process]]). For every square and move there is a list of (chance, next square, reward, finished).

   **`env.py`**
   ```python
   H, W = 4, 12; START, GOAL = (3, 0), (3, 11)          # cliff walking: the cliff is the bottom row between them
   MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]            # up, right, down, left
   sid = lambda r, c: r * W + c
   LIVE = [sid(r, c) for r in range(H) for c in range(W) if not (r == 3 and 0 < c < 11) and (r, c) != GOAL]   # 37 states

   def step_det(s, a):
       """one exact move -> (next state, reward, done). Every step costs 1; the cliff costs 100 and sends you back to the start"""
       r, c = divmod(s, W); dr, dc = MOVES[a]
       r, c = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
       if r == 3 and 0 < c < 11: return sid(*START), -100.0, False
       return sid(r, c), -1.0, (r, c) == GOAL

   def transitions(s, a, slip):
       """with chance `slip` the move is replaced by a random one of the 4; returns [(probability, next, reward, done)]"""
       return [(p,) + step_det(s, b) for b in range(4) if (p := (1 - slip) * (b == a) + slip / 4) > 0]
   ```

2. **Solve it exactly with value iteration, and check the answer three ways** ([[Dynamic Programming]], [[Bellman Equation]]). Value iteration starts every square at 0 and keeps replacing each square's value by the best move's expected value, until nothing changes by more than 1e-10. The three checks: (a) solve the best policy's linear equations directly, (b) check the Bellman equation holds, (c) simulate the policy many times ([[Monte Carlo]]).

   **`vi.py`**
   ```python
   import numpy as np
   from env import *

   def q_from(V, slip):
       Q = np.zeros((H * W, 4))
       for s in LIVE:
           for a in range(4): Q[s, a] = sum(p * (r + (0 if d else V[s2])) for p, s2, r, d in transitions(s, a, slip))
       return Q

   def value_iteration(slip, tol=1e-10):
       V, sweeps = np.zeros(H * W), 0
       while True:
           new = V.copy(); new[LIVE] = q_from(V, slip)[LIVE].max(1); sweeps += 1
           done = np.abs(new - V).max() < tol; V = new
           if done: return V, q_from(V, slip), sweeps

   def evaluate_exact(policy, slip):
       """exact value of a fixed policy by solving the linear Bellman equations"""
       idx = {s: i for i, s in enumerate(LIVE)}; A, b = np.eye(len(LIVE)), np.zeros(len(LIVE))
       for s in LIVE:
           for p, s2, r, d in transitions(s, policy[s], slip):
               b[idx[s]] += p * r
               if not d: A[idx[s], idx[s2]] -= p
       return dict(zip(LIVE, np.linalg.solve(A, b)))

   def rollout(policy_fn, slip, rng, cap=500):
       s, G = sid(*START), 0.0
       for _ in range(cap):
           b = rng.randint(4) if rng.rand() < slip else policy_fn(s)
           s, r, d = step_det(s, b); G += r
           if d: break
       return G

   def path_rows(Q, slip):
       """rows visited by the greedy policy when no slips happen"""
       s, rows = sid(*START), [3]
       for _ in range(60):
           s, _, d = step_det(s, int(Q[s].argmax())); rows.append(s // W)
           if d: break
       return rows

   if __name__ == "__main__":
       for slip in (0.0, 0.2):
           V, Q, sweeps = value_iteration(slip); start = sid(*START); pol = {s: int(Q[s].argmax()) for s in LIVE}
           exact = evaluate_exact(pol, slip); gap = max(abs(exact[s] - V[s]) for s in LIVE)
           resid = np.abs(V[LIVE] - Q[LIVE].max(1)).max()
           rng = np.random.RandomState(0); G = [rollout(lambda s: pol[s], slip, rng) for _ in range(20000)]
           print(f"slip {slip}: {sweeps} sweeps | V*(start) = {V[start]:.4f} | greedy policy solved exactly differs by {gap:.1e} | Bellman residual {resid:.1e}")
           print(f"   20,000 simulated episodes of that policy: mean return {np.mean(G):.3f} +- {np.std(G) / np.sqrt(len(G)):.3f} (standard error) | path rows without slips {path_rows(Q, slip)}")
   ```
   Run `python vi.py`. I saw:
   ```
   slip 0.0: 15 sweeps | V*(start) = -13.0000 | greedy policy solved exactly differs by 0.0e+00 | Bellman residual 0.0e+00
      20,000 simulated episodes of that policy: mean return -13.000 +- 0.000 (standard error) | path rows without slips [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
   slip 0.2: 119 sweeps | V*(start) = -28.6338 | greedy policy solved exactly differs by 3.2e-10 | Bellman residual 6.5e-11
      20,000 simulated episodes of that policy: mean return -28.417 +- 0.204 (standard error) | path rows without slips [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
   ```
   What to look for: with no slipping the answer is exactly −13 (the shortest route along the cliff edge is 13 steps). With slipping 0.2 the best route moves to the top row, because a random move beside the cliff is now expensive, and the value drops to −28.634. The simulated mean −28.417 ± 0.204 is within about one standard error of the exact value ([[Law of Large Numbers]]).

3. **Let Q-learning learn it by trial and error, and compare with the exact answer.** The Q-table has 48 × 4 cells. Each step nudges one cell toward "reward plus the best value of where I landed". Each setting is run on 5 seeds. The columns are: **start error** = how far the learned best value of the start square is from the exact one; **near-optimal moves** = share of the 37 squares where the exact value of the learned greedy move is within 0.5 of the exact value of the best move; **greedy-policy return** = the average score of the learned greedy policy over 1,000 test episodes (in the same slippery world, but with no random exploration); **online return** = the average score over the last 100 training episodes, with exploration.

   **`qlearn.py`**
   ```python
   import numpy as np
   from env import *
   from vi import value_iteration, rollout, path_rows

   def q_learning(slip, episodes, alpha, eps, seed, cap=500):
       rng = np.random.RandomState(seed); Q = np.zeros((H * W, 4)); returns = []
       for _ in range(episodes):
           s, G = sid(*START), 0.0
           for _ in range(cap):
               q = Q[s]; best = np.flatnonzero(q == q.max()); a = rng.randint(4) if rng.rand() < eps else int(best[rng.randint(len(best))])   # epsilon-greedy, random ties
               b = rng.randint(4) if rng.rand() < slip else a                                                     # the slip
               s2, r, d = step_det(s, b); G += r
               Q[s, a] += alpha * (r + (0 if d else Q[s2].max()) - Q[s, a])                                       # the Q-learning update
               s = s2
               if d: break
           returns.append(G)
       return Q, returns

   CONFIGS = {   # name: (slip, episodes, alpha, epsilon); one change at a time from the baseline
       "baseline":        (0.2, 1000, 0.5, 0.1),
       "epsilon 0.01":    (0.2, 1000, 0.5, 0.01),
       "epsilon 0.3":     (0.2, 1000, 0.5, 0.3),
       "alpha 0.05":      (0.2, 1000, 0.05, 0.1),
       "alpha 1.0":       (0.2, 1000, 1.0, 0.1),
       "100 episodes":    (0.2, 100, 0.5, 0.1),
       "5000 episodes":   (0.2, 5000, 0.5, 0.1),
       "no slip":         (0.0, 1000, 0.5, 0.1),
   }
   if __name__ == "__main__":
       start = sid(*START); exact = {sl: value_iteration(sl) for sl in (0.0, 0.2)}
       for name, (slip, n, alpha, eps) in CONFIGS.items():
           V, Qs, _ = exact[slip]; rows = []
           for seed in range(5):
               Q, ret = q_learning(slip, n, alpha, eps, seed)
               tie = np.random.RandomState(seed); pol = [int(tie.choice(np.flatnonzero(q == q.max()))) for q in Q]   # greedy policy, ties broken once
               greedy = lambda s: pol[s]
               ev = np.random.RandomState(100 + seed); gret = np.mean([rollout(greedy, slip, ev) for _ in range(1000)])
               near = np.mean([Qs[s, int(Q[s].argmax())] >= V[s] - 0.5 for s in LIVE])
               rows.append((abs(Q[start].max() - V[start]), near, gret, np.mean(ret[-100:])))
           m = np.mean(rows, 0)
           print(f"{name:14s} | exact V*(start) {V[start]:7.2f} | start error {m[0]:6.2f} | near-optimal moves {m[1]*100:5.1f}% | greedy-policy return {m[2]:7.2f} | online return, last 100 episodes {m[3]:8.2f} | greedy return per seed {np.round([r[2] for r in rows], 1).tolist()}")
   ```
   Run `python qlearn.py`. I saw:
   ```
   baseline       | exact V*(start)  -28.63 | start error   5.82 | near-optimal moves  76.8% | greedy-policy return  -42.67 | online return, last 100 episodes   -64.93 | greedy return per seed [-90.5, -35.0, -28.3, -28.8, -30.7]
   epsilon 0.01   | exact V*(start)  -28.63 | start error   7.21 | near-optimal moves  68.1% | greedy-policy return -120.87 | online return, last 100 episodes   -40.32 | greedy return per seed [-427.8, -59.6, -31.5, -30.0, -55.4]
   epsilon 0.3    | exact V*(start)  -28.63 | start error   5.17 | near-optimal moves  72.4% | greedy-policy return -109.58 | online return, last 100 episodes  -120.76 | greedy return per seed [-34.9, -31.9, -410.1, -40.7, -30.4]
   alpha 0.05     | exact V*(start)  -28.63 | start error   1.94 | near-optimal moves  81.1% | greedy-policy return  -29.67 | online return, last 100 episodes   -36.96 | greedy return per seed [-28.6, -28.6, -29.5, -29.5, -32.2]
   alpha 1.0      | exact V*(start)  -28.63 | start error   7.18 | near-optimal moves  66.5% | greedy-policy return -129.39 | online return, last 100 episodes  -120.20 | greedy return per seed [-232.1, -161.7, -101.8, -74.4, -76.9]
   100 episodes   | exact V*(start)  -28.63 | start error  11.82 | near-optimal moves  74.6% | greedy-policy return  -39.17 | online return, last 100 episodes  -160.98 | greedy return per seed [-51.0, -37.6, -28.8, -42.7, -35.8]
   5000 episodes  | exact V*(start)  -28.63 | start error   4.57 | near-optimal moves  77.3% | greedy-policy return  -29.47 | online return, last 100 episodes   -49.07 | greedy return per seed [-28.0, -29.1, -30.7, -29.7, -29.9]
   no slip        | exact V*(start)  -13.00 | start error   0.00 | near-optimal moves  97.3% | greedy-policy return  -13.00 | online return, last 100 episodes   -50.16 | greedy return per seed [-13.0, -13.0, -13.0, -13.0, -13.0]
   ```

## What the numbers say
- **Q-learning reaches the exact answer when nothing is random:** −13.00 on all 5 seeds and a start error of 0.00, and the learned values match the exact table on the start square. The 97.3% near-optimal moves, not 100%, is probably squares the agent rarely or never visited, but I did not check this.
- **Randomness makes the learning noisy, and the learning rate matters most:** each update jumps toward one random outcome, and a large jump keeps knocking the estimate around. That fits the picture in [[Monte Carlo]] and [[Law of Large Numbers]], where averaging many samples is what gives the exact answer, and a small step size does more averaging. In the data: learning rate 0.05 gave a start error of 1.94 and a greedy score of −29.67, 0.5 gave 5.82 and −42.67, and 1.0 gave 7.18 and −129.39. I only varied this in the slippery world, so I cannot say it also matters with no slipping.
- **More episodes helped:** 100 episodes gave −39.17, 1,000 gave −42.67 (with one bad seed), 5,000 gave −29.47 and all five seeds were within −28.0 to −30.7. The seed at −28.0 is above the exact best (−28.63) only because 1,000 simulated test episodes are noisy (the standard error is about 0.9: the 20,000-episode figure of 0.204 times the square root of 20).
- **Values converge more slowly than the policy:** after 5,000 episodes the start error was still 4.57, while the greedy policy was already close to optimal (−29.47 against −28.63). So the policy can be good while the numbers in the table are still off. I did not test why.
- **The means for the random-move rate are driven by one seed each:** ε 0.01 had one seed at −427.8 and ε 0.3 had one at −410.1. A score that low suggests a policy that loops or keeps falling off until the 500-step cap, but I did not inspect those runs. With 5 seeds, I cannot say whether 0.1 is better than 0.01 or 0.3.
- **Online return is always worse than the greedy return** (baseline −64.93 against −42.67, and with no slipping −50.16 against −13.00), because while learning the agent explores and falls off the cliff. That is the same effect as in the [[Q-Learning]] note.

## What went wrong, and the fix
- **The first version of `qlearn.py` was far too slow** and printed nothing for more than 7 minutes, because the output was buffered and the code built a new random generator for every greedy step. I stopped it and fixed two things: ties are broken once per run, and the training loop breaks ties by indexing instead of calling a slower function. I also ran it with unbuffered output (`python -u`). The results of the fixed version were identical across two runs.
- **A low-looking mean hid single bad seeds.** I added the per-seed greedy returns to the output after the first fixed run, which is how the −427.8 and −410.1 seeds were found.
- **A leftover expression in my first draft,** `0.0 if False else slip`, always evaluated to `slip`. I replaced it with plain `slip`, so the greedy test runs in the same slippery world as the training.

## What I did not run
- **A fresh virtual environment.** I used my existing environment (versions above). The rerun from this note used a fresh folder.
- **More than one rerun.** I extracted the three code blocks from this note into a new folder and ran `vi.py` and `qlearn.py` there. Both printed output identical to the first run (the runs are seeded and use NumPy only). That is a single rerun, and each setting has only 5 seeds.
- **A decaying learning rate, a decaying exploration rate, SARSA, double Q-learning,** a larger grid, or a different discount (the code uses no discount, 1.0).
- **Why the bad seeds failed,** and which squares were never visited.
- **The note in Obsidian.** I did not open it there.

## Concepts used
- [[Reinforcement Learning]] · [[Markov Decision Process]] · [[Bellman Equation]] · [[Dynamic Programming]] · [[Q-Learning]] · [[Monte Carlo]] · [[Law of Large Numbers]] · [[Learning Rate]] · [[Expected Value]] · [[Variance]] · [[NumPy]]
- [[Snake DQN]] · [[Deep Q-Network]]
- [[Looking Inside Snake DQN]]
## Next ideas
- Make the learning rate shrink over time and compare it with the fixed 0.05.
- Use 20 seeds, so the random-move-rate result has real error bars, and print which squares were never visited.
- Add SARSA and compare it with Q-learning in the slippery world.
- Replace the table with a small network (deep Q-learning) on a bigger grid, as in [[Q-Learning]]. A first network version, on Snake, is [[Snake DQN]].

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-rl What were the five "done" checks of the Q-Learning Grid World?::(1) the world is written as a table of transitions as in an MDP, (2) value iteration finds the best value of every square and is checked three ways, (3) Q-learning reaches the exact answer with no randomness, (4) in a slippery version eight settings are compared on 5 seeds each against the exact answer, (5) what is and is not shown is written down.

#flashcards/projects/concept #flashcards/topic/project-rl Describe the grid world.::4 rows by 12 columns. Start bottom-left, goal bottom-right. Each step costs 1. The 10 squares between them on the bottom row are a cliff: stepping on one costs 100 and sends you back to the start. That leaves 37 squares to stand on, not counting the goal. Four moves: up, right, down, left. In the slippery version, with chance 0.2 the move is replaced by a random one of the four.

#flashcards/projects/concept #flashcards/topic/project-rl What is the exact answer from value iteration?::No slipping: best total score from the start is −13.000 after 15 sweeps, along the row beside the cliff. Slipping 0.2: −28.634 after 119 sweeps, and the best route switches to the top row, away from the cliff.

#flashcards/projects/concept #flashcards/topic/project-rl How was the exact answer checked?::Solving the best policy's equations directly differs by 3.2e-10 and the equations hold to 6.5e-11. 20,000 simulated episodes of that policy averaged −28.417 with a standard error of 0.204, about one standard error from −28.634.

#flashcards/projects/concept #flashcards/topic/project-rl What did Q-learning do with no slipping?::It found the exact optimum on all 5 seeds: greedy-policy score −13.00 each time, with a start error of 0.00.

#flashcards/projects/example #flashcards/topic/project-rl What did Q-learning score in the slippery world (1,000 episodes, learning rate 0.5, 10% random moves)?::A greedy-policy score of −42.67 on average against the exact −28.63. Four seeds were between −28.3 and −35.0, and one seed scored −90.5.

#flashcards/projects/concept #flashcards/topic/project-rl What helped most in the slippery world?::A small learning rate. 0.05 gave −29.67 (all five seeds between −28.6 and −32.2), and 1.0 gave −129.39 (all five worse than −74). 5,000 episodes also helped (−29.47). The note ties this to Monte Carlo and the law of large numbers: a small step size does more averaging.

#flashcards/projects/concept #flashcards/topic/project-rl Why can the exploration-rate results not be ranked?::The means for 1% (−120.87) and 30% (−109.58) each come from a single failed seed (−427.8 and −410.1), against −42.67 at 10%. With 5 seeds the note cannot rank them, and it did not inspect why those seeds failed.

#flashcards/projects/example #flashcards/topic/project-rl Why was one seed's score above the exact best (−28.0 against −28.63)?::Only because the 1,000 simulated test episodes are noisy (standard error about 0.9, which is 0.204 times the square root of 20).

#flashcards/projects/concept #flashcards/topic/project-rl Why is online return always worse than the greedy return?::While learning the agent explores and falls off the cliff. Baseline −64.93 against −42.67, and with no slipping −50.16 against −13.00. The same effect as in the Q-Learning note.

#flashcards/projects/cloze #flashcards/topic/project-rl Values converge more slowly than the ==policy==: after 5,000 episodes the start error was still 4.57 while the greedy policy was already close to optimal (−29.47).

#flashcards/projects/concept #flashcards/topic/project-rl What was wrong with the first `qlearn.py`?::It was far too slow and printed nothing for over 7 minutes, because the output was buffered and the code built a new random generator for every greedy step. The fix: break ties once per run, index instead of calling a slower function, and run with `python -u`. Results were identical across two runs.

#flashcards/projects/concept #flashcards/topic/project-rl How were single bad seeds found?::The mean looked low but hid them. The note added per-seed greedy returns to the output, which showed the −427.8 and −410.1 seeds.

#flashcards/projects/concept #flashcards/topic/project-rl What did the Q-Learning Grid World not run?::A fresh virtual environment, a decaying learning or exploration rate, SARSA, double Q-learning, a larger grid, a discount other than 1.0, why the bad seeds failed, and more than one rerun. Each setting has only 5 seeds.
