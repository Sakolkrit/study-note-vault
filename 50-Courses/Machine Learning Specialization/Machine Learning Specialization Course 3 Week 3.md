---
tags: [lecture, course, machine-learning, reinforcement-learning, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 3, Week 3
lecturer: Andrew Ng
term: self-paced
source: one small lab notebook (the Mars rover state-action values) and the repo README; no lecture video; no notes file
status: seed
---

# Machine Learning Specialization, Course 3, Week 3: reinforcement learning

> **In one line:** A program that learns by trying actions and collecting rewards; the one worked example here is a rover on six squares, with a value for every state-action pair. (The title is my own label.)

## What the sources cover
1. **The README's list:** reinforcement learning introduction, the state-action value function, continuous state spaces. No notes file exists for this week.
2. **The lab (notebook).** A "Mars rover" on six states in a row. The two end states are terminal, with rewards (in the lab's default settings) of 100 on the left and 40 on the right; every other step earns 0. The lab has five cells and one call, `generate_visualization(...)`, which draws the values for the settings you choose: terminal rewards, a reward per step, a discount factor (default 0.5) and a probability of stepping the wrong way (default 0). See [[Reinforcement Learning]] and [[Q-Learning]].
3. **My own computation (not from the lab).** Because the lab only draws a picture and I could not run it, I computed the same example myself by repeatedly applying "reward now plus the discount times the best value of where you land". With the defaults the best return from states 2 to 5 is 50, 25, 12.5 and 20, the rover should go left from states 2 to 4 and right from state 5, and the table of values is below.

| State | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Q(s, left) | 100 | 50 | 25 | 12.5 | 6.25 | 40 |
| Q(s, right) | 100 | 12.5 | 6.25 | 10 | 20 | 40 |
| best | 100 | 50 | 25 | 12.5 | 20 | 40 |

(States 1 and 6 are terminal, so the return there is just their reward.)

## Key ideas to remember
- An agent picks actions, gets rewards, and wants the largest total reward over time ([[Reinforcement Learning]]).
- The value of taking an action in a state is the reward now plus the discounted value of the best thing you can do afterwards ([[Q-Learning]]).
- A discount factor below 1 makes near rewards count more than far ones: here 40 close by (from state 5) beats 100 far away.
- Which way to go can flip as rewards or the discount factor change.

## Papers named in the sources
- None.

## Questions I still have
- What does the lecture say about continuous states (the README's third topic)? I have no source for it.
- What does the misstep probability do to the table? The lab can show it; I did not compute that case.

## What this note is built from
- **No lecture video, no transcript, no notes file.** The lab notebook (5 cells) and the repo README only, so this week is the thinnest in the vault.
- **The lab notebook could not be rerun.** It imports `generate_visualization` from `libs/utils.py`, but that function is not in the repo's copy of the file, so the scratch run failed with a name error. The table above is my own computation, not the lab's output.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 3 Week 2]]. Lecture after: none.
- Concept notes this week feeds: [[Reinforcement Learning]], [[Q-Learning]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-3 ML Specialization Course 3 Week 3: what does it cover?::Reinforcement learning: the README's list (introduction, state-action value function, continuous state spaces) and one lab, a "Mars rover" on six states.

#flashcards/courses/concept #flashcards/topic/course-mls-3 What is the Mars rover example?::Six states in a row. The two end states are terminal with rewards (lab defaults) of 100 on the left and 40 on the right. Every other step earns 0. The lab draws values for chosen terminal rewards, a reward per step, a discount factor (default 0.5) and a probability of stepping the wrong way (default 0).

#flashcards/courses/concept #flashcards/topic/course-mls-3 What are the rover's state-action values with the defaults? (the note's own computation)::Q(left): 100, 50, 25, 12.5, 6.25, 40 for states 1 to 6. Q(right): 100, 12.5, 6.25, 10, 20, 40. Best: 100, 50, 25, 12.5, 20, 40. States 1 and 6 are terminal. (A recomputation from "reward now plus discount times the best value of where you land" gives the same table.)

#flashcards/courses/concept #flashcards/topic/course-mls-3 What should the rover do in each state?::Go left from states 2 to 4 and right from state 5, because 40 close by (from state 5) beats 100 far away with a discount of 0.5.

#flashcards/courses/concept #flashcards/topic/course-mls-3 What is the value of an action in a state?::The reward now plus the discounted value of the best thing you can do afterwards.

#flashcards/courses/concept #flashcards/topic/course-mls-3 Why can the best direction flip?::It depends on the terminal rewards and the discount factor.

#flashcards/courses/concept #flashcards/topic/course-mls-3 Why is Week 3 the thinnest week?::No lecture video, no transcript and no notes file. The lab notebook (5 cells) and the README only. The notebook could not be rerun, because it imports `generate_visualization` from `libs/utils.py`, and the repo's copy of the file does not contain that function. The table is the note's own computation, not the lab's output. The misstep-probability case was not computed.
