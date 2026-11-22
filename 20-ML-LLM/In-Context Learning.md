---
tags: [LLM, prompting, in-context-learning, few-shot, zero-shot, GPT-3]
status: growing
level: 4
tier: 2
---

# In-Context Learning

Often shortened to **ICL**. A model picks up a new task **from examples written in the prompt**, with none of its weights changed.

## 🟢 1. Simple (age 12)
Imagine showing a smart helper a few solved examples: "cat → chat, dog → chien, bear → ?" without any lesson, and they work out you want French. In-context learning is the AI doing that: you **put examples in the question itself**, and the AI copies the pattern. It does not "study" anything. Once the chat ends, it has not changed.

**Example:** Prompt: "Q: How old is this bear? A: 4. Q: How old will the bear be next year? A:" The example shows the format, and the model answers "5".

## 🔵 2. How it works
1. A large [[LLM]] is trained to continue text. If the text is a pattern of examples, the best continuation follows the pattern.
2. **Zero-shot:** you just ask. The question has no examples. How well it does depends on the model.
3. **Few-shot:** you add a few input and output examples before your question. This usually works better.
4. The model's numbers (weights) do **not** change. Everything it "learns" lives in the prompt and disappears afterwards.
5. The costs of showing examples (lecture): it needs effort to write them, it makes the prompt longer so computing costs more, and it adds delay.
6. Prompts have a typical structure: context, instructions, the input, and constraints ([[Prompt Engineering]]).
7. Related tricks: explain the steps ([[Chain of Thought]]), take a vote of several answers, and watch the length: the lecture warns of "context rot", where very long inputs can hurt performance (it cites a 2025 study I did not read).
8. If you need the behaviour every time without long prompts, you can bake it in by [[Instruction Tuning]] or [[Fine-tuning]].

## 🟠 3. Technical detail
- **Where the term comes from:** the GPT-3 paper, *Language Models are Few-Shot Learners* (Brown et al., 2020). Its abstract says that scaling up a language model greatly improves few-shot performance, with GPT-3 (175 billion parameters) applied "without any gradient updates or fine-tuning", with tasks and examples given purely as text (arXiv 2005.14165, read).
- **Zero-shot against few-shot (lecture):** zero-shot depends heavily on the base model's performance, and few-shot typically does better.
- **What "learning" means here:** the model computes a different answer because its input is different. Whether this is *really* learning inside the network is a research question, and I do not claim an answer.
- **The code's analogy:** a learner with fixed settings that weights each prompt example by how close its input is to the new input (a bit like attention). Its error falls as more examples are included. This is an illustration of "only the prompt changes", not a model of how a Transformer does it.
- **Limits:** examples take up context, order and choice of examples can matter, and the model may copy surface patterns (the last two are general experience, not tested here).
- **Cost view:** every example is paid for again in every request.

```python
import numpy as np
rng = np.random.RandomState(0)
f = lambda x: np.sin(2 * x)                                     # the hidden rule the "prompt examples" follow
def predict(x_new, xs, ys, width=0.3):
    """a stand-in learner with FIXED settings: weight each example by how close its input is to the new input, then average its outputs
    (this is attention-like, but it is only an analogy, not a claim about how real LLMs work)"""
    w = np.exp(-((xs - x_new) ** 2) / (2 * width ** 2)); return float((w * ys).sum() / w.sum())
test = np.linspace(-1.5, 1.5, 41)
print('examples in the prompt | mean error on 41 new inputs (nothing is trained; only the prompt changes)')
for k in (1, 2, 4, 8, 16, 64):
    errs = []
    for _ in range(200):
        xs = rng.uniform(-1.5, 1.5, k); ys = f(xs)
        errs.append(np.mean([abs(predict(t, xs, ys) - f(t)) for t in test]))
    print(f'{k:5d} | {np.mean(errs):.3f}')
# the cost side: a few-shot prompt is longer than a zero-shot prompt
question = 'Q: How old will the bear be next year ?'.split()
example = 'Q: How old is this bear ? A: 4 .'.split()
for k in (0, 1, 5, 20):
    n = len(question) + k * len(example)
    print(f'{k:2d} examples -> prompt of {n:3d} words | attention score cells {n * n:6d}')
```

## 🔴 4. Deep / formal
**A fixed-setting learner that uses the prompt.** The code's stand-in predicts a new output as a weighted average of the example outputs:

$$
\hat y(x)=\frac{\sum_{i=1}^{k}w_i\,y_i}{\sum_{i=1}^{k}w_i},\qquad w_i=\exp\!\left(-\frac{(x-x_i)^2}{2h^2}\right)
$$

**Where:**
- $(x_i,y_i)$ are the $k$ examples in the prompt, and $x$ is the new input.
- $w_i$ is how much example $i$ counts: close to 1 when $x_i$ is near $x$, near 0 when it is far away.
- $h$ is a fixed width that sets what "near" means (0.3 in the code).
- $\hat y(x)$ is the prediction. **Matters:** nothing here is trained: with the same settings, different examples give a different function. More examples fill in the gaps, so the prediction improves. This is only an analogy for ICL (the weights look like attention scores), and real models are far more complicated.

**Example (tiny numbers):** the hidden rule is $y=\sin(2x)$. Two examples at $x_1=-0.3$ and $x_2=0.3$, so $y_1=-0.565$ and $y_2=+0.565$. New input $x=0$, $h=0.3$.
- Both examples are the same distance (0.3) from 0, so $w_1=w_2=e^{-0.5}=0.607$.
- $\hat y=(0.607\times(-0.565)+0.607\times0.565)/(0.607+0.607)=\mathbf{0}$.
- The true value is $\sin(0)=0$.
- Meaning: examples on both sides pinned the answer down. In the code, the mean error over 41 test inputs falls from $0.818$ with 1 example to $0.333$ with 4, $0.173$ with 16 and $0.141$ with 64 (the last drop is small, because a smoother of fixed width has its own limit).

**Prompt cost.** Attention compares every token with every other, so for a prompt of $n$ tokens:

$$
\text{score cells}\;\propto\;n^{2}\quad\text{and}\quad n=n_{\text{question}}+k\cdot n_{\text{example}}
$$

**Where:**
- $n_{\text{question}}$ is the length of your question and $n_{\text{example}}$ is the length of one example, $k$ is how many examples.
- **Matters:** each example adds length, and length is paid for quadratically in the attention grid (before tricks such as the [[KV Cache]] and [[FlashAttention]]). So 20 examples are far more than 20 times the grid of one.

**Example (tiny numbers):** a 10-word question and examples of 10 words each.
- 0 examples: $n=10$, cells $=\mathbf{100}$.
- 5 examples: $n=60$, cells $=\mathbf{3{,}600}$.
- 20 examples: $n=210$, cells $=\mathbf{44{,}100}$.
- Meaning: 20 examples make the prompt 21 times longer and the attention grid 441 times bigger. Real systems reduce some of this cost, but not the extra length.

**Further notes:**
- **The analogy's limits:** the smoother only works on numbers with a smooth rule. Real ICL works on words, formats and tasks.
- **Not covered:** why ICL emerges with scale, and prompt caching (which the lecture's retrieval part uses to cut costs).

**What came before, and how this improved on it:**
- **Before:** pretrain and then fine-tune a separate model for each task, which, the GPT-3 abstract says, needs task-specific datasets of thousands or tens of thousands of examples.
- **Its limit:** humans can do a new language task from a few examples or simple instructions, but such systems struggled to (GPT-3 abstract).
- **What changed:** scale the model and give a few examples as text in the prompt, with no weight updates (Brown et al., arXiv 2005.14165, 28 May 2020, abstract read).
- **How much better:** the abstract says few-shot GPT-3 sometimes reaches competitiveness with prior fine-tuned state of the art, with no fine-tuning. I did not check any figures. In this note's code, a fixed-setting learner's error fell from $0.818$ to $0.173$ as examples went from 1 to 16, and 20 examples made the prompt 21 times longer.

## Related
- [[Prompt Engineering]] · [[LLM]] · [[Chain of Thought]] · [[Instruction Tuning]] · [[Fine-tuning]] · [[Attention]] · [[Self-Attention]] · [[KV Cache]] · [[Transformer]] · [[GPT-3 2020]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 3 slides, Afshine and Shervine Amidi
- Brown et al., *Language Models are Few-Shot Learners*, 2020 (arXiv 2005.14165, abstract read)
- Hong et al., *Context Rot: How Increasing Input Tokens Impacts LLM Performance*, 2025 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm In-context learning:::Put a few solved examples in the question itself and the AI copies the pattern. It does not study anything and has not changed once the chat ends.

#flashcards/ml/formula #flashcards/topic/llm Write the kernel-weighted prediction used as a toy model of in-context learning, and explain each symbol.
?
$$\hat y(x)=\frac{\sum_{i=1}^{k}w_i\,y_i}{\sum_{i=1}^{k}w_i},\qquad w_i=\exp\!\left(-\frac{(x-x_i)^2}{2h^2}\right)$$
- $(x_i,y_i)$: the $k$ examples in the prompt. $x$: the new input.
- $w_i$: how much example $i$ counts. Close to 1 when $x_i$ is near $x$, near 0 when far away.
- $h$: a fixed width that sets what "near" means (0.3 in the code).
- $\hat y(x)$: the prediction, a weighted average of the example answers. Nothing is trained.

#flashcards/ml/formula #flashcards/topic/llm Write how the prompt length affects attention cost, and explain each symbol.
?
$$\text{score cells}\;\propto\;n^{2}\quad\text{and}\quad n=n_{\text{question}}+k\cdot n_{\text{example}}$$
- $n_{\text{question}}$: the length of your question. $n_{\text{example}}$: the length of one example. $k$: how many examples.
- Each example adds length, and length is paid for quadratically in the attention grid.
- So 20 examples cost far more than 20 times as much.

#flashcards/ml/cloze #flashcards/topic/llm In-context learning changes the ==prompt==, not the model's ==weights==.

#flashcards/ml/example #flashcards/topic/llm The rule is $y=\sin(2x)$. Examples at $x=-0.3$ and $x=0.3$ ($y=\mp0.565$), new input $x=0$, $h=0.3$. What is $\hat y$?::Both are 0.3 away, so w₁ = w₂ = e^−0.5 = 0.607. ŷ = (0.607 × (−0.565) + 0.607 × 0.565) / 1.214 = 0. The true value sin 0 = 0.

#flashcards/ml/example #flashcards/topic/llm A 10-word question and examples of 10 words each. How many score cells for 0, 5 and 20 examples?::0 examples: n = 10, 100 cells. 5: n = 60, 3,600. 20: n = 210, 44,100. The prompt is 21 times longer and the grid 441 times bigger.
