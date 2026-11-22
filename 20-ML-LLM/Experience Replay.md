---
tags: [experience-replay, replay-buffer, DQN, reinforcement-learning, off-policy, ML]
status: growing
level: 4
tier: 2
---

# Experience Replay

Experience replay is **a memory of past moves that a learning agent practises from, again and again, in random order**. Instead of learning only from what just happened, the agent stores each step in a buffer and, at every update, learns from a random handful of old steps. It is one of the two helpers of a [[Deep Q-Network]].

## 🟢 1. Simple (age 12)
Think of a student who has a box of flash cards. Each time something happens in the game, a new card goes into the box: "I was here, I did this, this is what came out." To study, the student does not read the last card ten times in a row. They shuffle the box and pull out a few cards at random. Old cards get read again and again, and the cards come in a mixed order.

**Example:** a Snake player has the cards "wall ahead, I turned left, I lived", "food on the right, I went right, I ate it" and thousands more. Each study round uses 64 random cards, so one good lesson is used many times.

## 🔵 2. How it works
1. **Play** one step: see the state, choose a move ([[Epsilon-Greedy]]), get a reward and a new state.
2. **Store** the step as one record: (state, move, reward, next state, game over?).
3. **Keep a fixed size.** The buffer holds, say, the last 20,000 steps. When it is full, the oldest record is overwritten by the newest (like a circular tape).
4. **Wait** until the buffer has some records (1,000 in my project) so the first batches are not all the same few steps.
5. **Sample:** every update, pick $B$ random records (64 in my project).
6. **Learn** from that batch ([[Deep Q-Network]], [[Gradient Descent]]).
7. **Repeat.** Each record is picked many times before it is overwritten.

## 🟠 3. Technical detail
- **Why random order helps.** Steps that follow each other are very alike: after one step the snake is nearly in the same place, so the next sample says almost the same thing. A network trained on a long run of look-alikes drifts toward them and forgets other situations. Random draws mix the situations and give batches that look more like independent samples. (The usual explanation; I show the "alike" part below, but did not test that it is the cause of any failure.)
- **Why re-use helps.** Playing is slow and a gradient step is fast. Replay lets each played step feed many updates (sample efficiency).
- **The chance a stored record is in one batch.** With $n$ records in the buffer and a batch of $B$ drawn at random:

$$
P(\text{record is in a given batch})=\frac{B}{n}
$$

**Where:**
- $n$ is how many records are in the buffer right now (at most its capacity, 20,000 in my project).
- $B$ is the batch size (64).
- Draws are uniform and with replacement, so the chance is approximately $B/n$ (exactly $1-(1-1/n)^B$, which is about the same when $n$ is large).
- **Matters:** it shows both effects. A bigger buffer means each record is picked less often per update but stays longer; a bigger batch re-uses data more. With a very small buffer the batch is nearly the whole buffer and random order buys little.

**Example (tiny numbers):** a buffer of 6 records and batches of $B=2$ (made up; my arithmetic).
- Chance a given record is in one batch: $2/6=\mathbf{0.333}$.
- In 3 batches you would expect it about $3\times0.333=1$ time.
- Now my project's numbers: $n=20{,}000$ and $B=64$ give $64/20000=0.0032$ per update, yet a record stays about 20,000 updates, so it is expected in $20000\times0.0032=\mathbf{64}$ batches before it is overwritten.
- Meaning: one played step is re-used about 64 times.

- **The same count over a whole run (my arithmetic).** In [[Snake DQN]] there were 60,000 steps and the first update was at step 1,000, so 59,001 updates of 64 samples = 3,776,064 samples, about 62.9 uses per step on average. The version without replay used each step once (one sample per update).
- **Code: how alike are neighbouring steps?** A random player plays 5,000 steps in the Snake environment of [[Snake DQN]] (it needs that project's `snake.py` in the same folder). For each of the 8 features I compute the correlation between each step and the next one in time, and between two steps drawn at random from the buffer:

```python
# file: toy_replay.py
import random
import numpy as np
from snake import Snake

env, rng, obs = Snake(), random.Random(0), []
o = env.reset(seed=0)
for _ in range(5000):                                  # a random player; remember every observation in order
    obs.append(o); o, _, done = env.step(rng.randrange(3))
    if done: o = env.reset()
X = np.array(obs)
idx = np.random.default_rng(0).integers(len(X), size=(2, 5000))
print("feature              consecutive steps   two random steps")
for f, name in zip(range(3, 11), ["heading up", "heading right", "heading down", "heading left", "food is up", "food is right", "food is down", "food is left"]):
    cons = np.corrcoef(X[:-1, f], X[1:, f])[0, 1]      # each sample with the one right after it
    rand = np.corrcoef(X[idx[0], f], X[idx[1], f])[0, 1]   # each sample with a random other one from the buffer
    print(f"{name:<18}{cons:>14.3f}{rand:>19.3f}")
```

```text
feature              consecutive steps   two random steps
heading up                 0.084              0.009
heading right              0.061             -0.013
heading down               0.047              0.016
heading left               0.108              0.002
food is up                 0.807             -0.023
food is right              0.807              0.005
food is down               0.810             -0.013
food is left               0.799              0.001
```

- **Reading it:** the four "food is ..." features are about 0.8 correlated between consecutive steps and about 0 between random draws (all within $\pm0.03$). The "heading" features are much less alike in a row (0.05 to 0.11) because a random player turns often. So for the food direction, a run of consecutive steps carries little new information, and a random batch is much more varied. This shows the "alike" claim for one feature type and one random player; it does not show that replay fixes a training failure.
- **What the project measured.** After 5,000 steps (30 test games, mean of 5 runs) the full DQN scored 15.1 foods and the version without replay 4.9; the version with neither replay nor target network scored 9.4. By 60,000 steps all versions reached about 15.5. So replay sped up learning; it did not raise the final level in that game. I did not test whether the speed-up comes from the re-use (about 63 times) or from the random order.

## 🔴 4. Deep / formal
- **Off-policy needed.** The records were made by older versions of the agent, so learning from them only makes sense for *off-policy* methods such as [[Q-Learning]] and DQN. A method that must learn from its own current behaviour, such as [[PPO]], cannot reuse old records this simply.
- **Stale data.** An old record says what the *old* agent did and saw. A buffer that is too big holds steps from a much worse agent; one that is too small forgets rare situations and loses the benefit. The size is a setting, and I did not tune it.
- **Memory.** Each record stores two states. For image states (millions of steps of screens) the buffer is big, which is a real cost (from memory).
- **Uniform is not the only choice.** *Prioritised experience replay* picks steps with large errors more often (from memory, not run here). Uniform sampling treats a boring step and a rare, surprising one alike.
- **Terminal flag.** Each record must keep whether the game ended, or the goal in [[Deep Q-Network]] would count a future after the end.
- **Time limits.** In my project a game that went 100 steps without food was ended and stored as a normal "game over" record with reward $-1$. I did not separate a true end from a time-out.

**What came before, and how this improved on it:**
- **Before:** online learning: each step was used once, right after it happened, and then thrown away. Replay was proposed by Long-Ji Lin around 1992 and became standard through the Atari DQN papers of 2013 and 2015 (from memory, not checked).
- **Its limit:** consecutive steps are almost the same, so updates are highly correlated, and every experience is used only once, which wastes slow-to-collect data.
- **What changed:** store steps in a buffer and train on random batches from it.
- **How much better:** measured here, in Snake after 5,000 steps: 15.1 foods with replay against 4.9 without (30 test games, mean of 5 runs; the ranges were 14.1 to 16.6 and 0.7 to 16.1), and 63 uses per step on average against one. The final scores were equal. The published gains in Atari are from memory and I do not quote a figure.

## Related
- [[Deep Q-Network]] · [[Q-Learning]] · [[Epsilon-Greedy]] · [[Reinforcement Learning]] · [[Markov Decision Process]] · [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Policy Gradient]] · [[PPO]] · [[Snake DQN]] · [[Autocorrelation]]

## Sources
- My own toy (Python 3.9.6, NumPy 1.24.4) and the [[Snake DQN]] project. The history (Lin about 1992, the Atari papers of 2013 and 2015), prioritised replay and the memory cost for image states are from memory; I did not read any paper for this note.
- No separate picture: the explanatory chart of deep-RL tricks is in [[Deep Q-Network]].

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Experience replay:::A fixed-size memory of past steps (state, move, reward, next state, game over?). The agent learns from random batches drawn from it, so each step is re-used many times and the batch is varied.

#flashcards/ml/reverse #flashcards/topic/alignment-rl Replay buffer:::The memory in experience replay. When it is full the oldest record is overwritten by the newest.

#flashcards/ml/concept #flashcards/topic/alignment-rl What are the steps of experience replay?
?
Play a step; store the record (state, move, reward, next state, game over?); overwrite the oldest when full; wait until there are some records; draw a random batch; learn from it; repeat.

#flashcards/ml/concept #flashcards/topic/alignment-rl Why does experience replay help, and what is only the usual explanation?
?
Random draws mix very different situations (consecutive steps are alike) and each played step feeds many updates. That the alikeness is what breaks training without replay is the usual explanation; my toy only showed that consecutive steps are alike.

#flashcards/ml/formula #flashcards/topic/alignment-rl What is the chance a stored record is in one replay batch?
?
$$P(\text{record is in a given batch})=\frac{B}{n}$$
- $B$: batch size (64 in my project).
- $n$: how many records are in the buffer now (at most 20,000).
- Draws are uniform, with replacement, so this is approximate (exactly $1-(1-1/n)^B$).
- Matters: a bigger buffer lowers the chance per update but keeps records longer; a bigger batch re-uses data more.

#flashcards/ml/example #flashcards/topic/alignment-rl Buffer of 20,000 records and batches of 64: what is the chance per update, and how often is a record used before it is overwritten?::$64/20000=0.0032$ per update. A record stays about 20,000 updates, so it is expected in $20000\times0.0032=64$ batches, about 64 uses.

#flashcards/ml/concept #flashcards/topic/alignment-rl What did the replay toy measure, and what did it not show?
?
For a random Snake player, the "food is up, right, down, left" features were about 0.8 correlated between consecutive steps and about 0 between two random draws; the heading features were only 0.05 to 0.11. It showed consecutive steps are alike for that feature type; it did not show that replay fixes a failure.

#flashcards/ml/concept #flashcards/topic/alignment-rl What did the Snake project measure about replay?
?
After 5,000 steps: 15.1 foods with replay, 4.9 without, 9.4 with neither replay nor target network (30 test games, mean of 5 runs). By 60,000 steps all versions reached about 15.5. Replay sped up learning, not the final level; why was not tested.

#flashcards/ml/concept #flashcards/topic/alignment-rl Why must experience replay go with off-policy methods?
?
The stored records were made by older versions of the agent. Off-policy methods such as Q-learning and DQN can learn from them; methods that must learn from their current behaviour, such as PPO, cannot reuse old records this simply.

#flashcards/ml/concept #flashcards/topic/alignment-rl What are the risks of a buffer that is too big or too small?
?
Too big: it holds records from a much worse, older agent. Too small: it forgets rare situations and the batch is nearly the whole buffer, so random order buys little. The size is a setting; I did not tune it.

#flashcards/ml/concept #flashcards/topic/alignment-rl What came before experience replay and which facts are from memory?
?
Online learning: each step used once and then dropped. Replay was proposed by Long-Ji Lin around 1992 and made standard by the Atari DQN papers of 2013 and 2015. All of that, and prioritised replay, are from memory.

#flashcards/ml/cloze #flashcards/topic/alignment-rl In experience replay the oldest record is ==overwritten== when the buffer is full, and each update learns from a ==random batch==.
