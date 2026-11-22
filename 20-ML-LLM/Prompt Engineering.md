---
tags: [prompt-engineering, LLM, in-context-learning, few-shot, evaluation]
status: growing
level: 4
tier: 2
---

# Prompt Engineering

## 🟢 1. Simple (age 12)
Prompt engineering is **learning how to ask an AI so it gives you a good answer**. The text you type is called the **prompt**. A vague question gets a vague answer, and a clear question with a good example gets a much better one.

**Example:** Ask a new helper "make me something to eat" and you might get a bowl of plain rice. Ask "make me a cheese sandwich with no crusts, cut in triangles" and you get exactly what you pictured. The AI cannot read your mind, so the clearer and more specific you are, the better the result. Showing one finished sandwich as an example is even better.

## 🔵 2. How it works
1. **Say what you want clearly.** Who is the answer for, how long should it be, and what should it look like?
2. **Give background.** The AI knows nothing about your situation unless you tell it. Paste in the text, the facts or the rules it needs.
3. **Show examples.** Give two or three samples of a good question and answer. The AI copies the pattern. (This is called few-shot prompting.)
4. **Ask for the format.** For example "a list of 5 bullet points" or "answer in a table".
5. **Let it think in steps.** For tricky problems, ask it to work through the steps before the final answer.
6. **Try, look, adjust.** Your first prompt is rarely the best one. Change one thing at a time and compare the answers, like an experiment.

A prompt does not teach the AI new facts permanently or make it smarter. It only steers what the [[LLM]] does *right now*. To add facts, use [[RAG]]. To change its habits for good, use [[Fine-tuning]].

No image: I did not find a suitable free picture for this topic on Wikimedia Commons, so I left it out.

## 🟠 3. Technical detail
- **Why prompts work:** an [[LLM]] predicts the next token given everything before it. Your prompt is the "everything before it", so changing the prompt changes which continuations are likely. This is **in-context learning**: the model picks up a pattern from the text in front of it without any change to its weights.
- **Parts of a good prompt:**
  - **Role or setting:** "You are a careful maths tutor for 12-year-olds."
  - **Task:** exactly what to do.
  - **Context:** the facts, documents or data to use.
  - **Examples (few-shot):** input and output pairs. Zero-shot means no examples, one-shot means one.
  - **Output format:** a template, a list, or a data format such as JSON.
  - **Constraints:** length, tone, things to avoid, what to do if unsure ("say you don't know").
- **System prompt vs user prompt:** many systems let you set a standing instruction (the system prompt) that shapes every reply, separate from each question.
- **Delimiters:** wrap pasted text in clear markers (such as triple quotes or tags) so the model can tell your instructions from your data. This also helps against prompt injection (see [[Agent]]).
- **Reasoning prompts:** asking for step-by-step working ("chain of thought") often improves answers on maths and logic, and costs more tokens. Some modern models reason by default.
- **Sampling settings:** temperature and similar settings (see [[Softmax]]) control how random the output is. Low temperature gives steadier answers, high temperature more varied ones.
- **Self-consistency:** ask the same question several times and take the most common answer (see the maths below).
- **Context window:** the prompt plus the answer must fit into the model's limit, counted in tokens (see [[Tokenization]]).
- **Test your prompts:** keep a small set of example inputs with known good answers. Run every new version of the prompt on them. A prompt that looks better on one example may be worse overall.

```python
examples = [("The movie was great", "positive"), ("I wasted my money", "negative")]
def build(text):
    shots = "\n".join(f"Review: {r}\nLabel: {l}" for r, l in examples)
    return f"Label each review as positive or negative.\n\n{shots}\n\nReview: {text}\nLabel:"
print(build("Loved every minute"))
# Label each review as positive or negative.
#
# Review: The movie was great
# Label: positive
# Review: I wasted my money
# Label: negative
#
# Review: Loved every minute
# Label:

from math import comb
def majority(n, p):                           # chance a majority of n votes is right
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n // 2 + 1, n + 1))
print(round(majority(3, 0.7), 3), round(majority(5, 0.7), 3), round(majority(3, 0.4), 3))
# 0.784 0.837 0.352
```

## 🔴 4. Deep / formal
**Self-consistency: voting over several answers.** Ask the model the same question $n$ times (with some randomness), and take the most common answer. If each answer is independently right with chance $p$, the chance that a majority is right is:

$$
P(\text{majority right})=\sum_{k=\lfloor n/2\rfloor+1}^{n}\binom{n}{k}\,p^{k}(1-p)^{n-k}
$$

**Where:**
- $n$ is the number of answers collected (use an odd number so there are no ties), and $p$ is the chance that a single answer is right.
- $k$ is how many of the $n$ answers are right. The sum runs over every $k$ that makes a **majority**, from just over half up to all $n$.
- $\binom{n}{k}$ ("$n$ choose $k$") counts the different ways to pick which $k$ of the $n$ answers are the right ones.
- $p^{k}$ is the chance those $k$ are all right, and $(1-p)^{n-k}$ is the chance the other $n-k$ are all wrong.
- **Matters:** if $p>0.5$ the majority is more reliable than a single answer, and more votes help. If $p<0.5$ voting makes things *worse*, since the most common answer is then more likely to be wrong.
- **Caveat:** repeated answers from the same model are not truly independent (the same blind spots repeat), so real gains are smaller than the formula suggests.

**Example (tiny numbers):** $p=0.7$.
- $n=3$: right if 2 or 3 of the 3 are right. $\binom32\cdot0.7^2\cdot0.3+\binom33\cdot0.7^3=3\cdot0.49\cdot0.3+0.343=0.441+0.343=\mathbf{0.784}$.
- $n=5$: $\mathbf{0.837}$. So five votes beat three, which beat one answer ($0.70$).
- If instead $p=0.4$: $n=3$ gives $\mathbf{0.352}$, **lower** than $0.4$. Voting amplifies whichever side is more likely.
- Meaning: voting is a good trick when the model is usually right and a bad one when it is usually wrong, and it costs $n$ times as much.

**The token budget.** The prompt and the reply share the same limited space, the context window:

$$
n_{\text{prompt}}+n_{\text{reply}}\le n_{\text{context}}
$$

**Where:**
- $n_{\text{prompt}}$ is the number of tokens in everything you send: instructions, examples, pasted documents and the earlier conversation.
- $n_{\text{reply}}$ is the number of tokens the model writes back.
- $n_{\text{context}}$ is the model's context window, a fixed limit (see [[Tokenization]]).
- **Matters:** every example or document you add leaves less room for the answer, and a prompt that is too long gets cut or rejected. Long prompts also cost more and can dilute the important instructions.

**Example (tiny numbers):** a window of $8000$ tokens (an illustrative number; real models differ).
- Your prompt, with instructions, three examples and a pasted article, is $6500$ tokens.
- The most the reply can be: $8000-6500=\mathbf{1500}$ tokens.
- Using the common rule of thumb that a token is about $0.75$ of an English word (from memory, and it varies by language), that is about $1125$ words.
- Meaning: if you need a longer answer, trim the prompt (shorter examples, only the relevant part of the article).

**Further notes:**
- **Prompting vs training:** prompts change behaviour without changing weights, so they are cheap and reversible, but fragile: small wording changes can change results. [[Fine-tuning]] is more stable but costs data and compute.
- **Order and position effects:** models can pay more attention to the start and end of a long prompt than the middle. Put key instructions where they are not buried. (A commonly reported effect; behaviour varies by model.)
- **Prompt injection and leaks:** instructions hidden in pasted data can override yours, and a prompt can sometimes be coaxed out of a model. Do not put secrets in prompts.
- **Automatic prompt tuning:** some systems search for better prompts automatically, or tune a few soft "virtual tokens" instead of text (see [[Fine-tuning]]). An active research area.
- **Models change:** a prompt tuned for one model may behave differently on another, or after an update, so re-test.

**What came before, and how this improved on it:**
- **Before:** train or fine-tune a separate model for each task.
- **Its limit:** that needs labelled data and training for every new job.
- **What changed:** describe the task in the prompt, with examples. The chain-of-thought paper shows that adding worked reasoning steps to the examples helps: a 540B-parameter model prompted with just eight chain-of-thought examples reached state-of-the-art accuracy on the GSM8K maths word problems, surpassing a fine-tuned GPT-3 with a verifier (Wei et al., arXiv 2201.11903, 28 January 2022, abstract read).
- **How much better:** the published gain is above; I did not reproduce it. What the code below counts is the price: with assumed sizes (a 40-token question and 120 tokens per worked example), an 8-example prompt makes every call read about 1,000 tokens, 25 times more than the bare question.

**Check it yourself:**

```python
# Cost of a few-shot prompt, with assumed sizes (the real sizes depend on the task).
question, exemplar = 40, 120                 # tokens: the question, and one worked example with its reasoning
for k in (0, 1, 8):
    print(f"{k} worked examples in the prompt: about {question + k * exemplar} tokens read on every call")
print("8 examples vs none:", round((question + 8 * exemplar) / question, 1), "x the input tokens per call")
```

## Related
- [[LLM]] · [[Agent]] · [[RAG]] · [[Fine-tuning]] · [[RLHF]] · [[Tokenization]] · [[Softmax]] · [[Probability]] · [[Perplexity]] · [[Embedding Model]] · [[Decoding Strategies]] · [[Chain of Thought]] · [[In-Context Learning]] · [[Hallucination]] · [[Chat Template]] · [[Structured Output]] · [[System Prompt]] · [[Prompt Injection]]

## Sources
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3), 2020
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, 2022 (from memory)
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*, 2022 (from memory)
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, 2023 (from memory)
- Anthropic and OpenAI prompting guides (free online; check the current versions)
- Wei et al., arXiv 2201.11903 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Prompt engineering:::Learning how to ask an AI so it gives a good answer. A vague question gets a vague answer, and a clear question with a good example gets a much better one.

#flashcards/ml/formula #flashcards/topic/llm Write the chance a majority of $n$ answers is right and explain each symbol.
?
$$P(\text{majority right})=\sum_{k=\lfloor n/2\rfloor+1}^{n}\binom{n}{k}\,p^{k}(1-p)^{n-k}$$
- $n$: the number of answers collected (use an odd number so there are no ties). $p$: the chance a single answer is right.
- $k$: how many of the $n$ answers are right. The sum runs over every $k$ that makes a majority.
- $\binom{n}{k}$: "$n$ choose $k$", the number of ways to pick which answers are right.
- Voting amplifies: it helps when $p>0.5$ and hurts when $p<0.5$.

#flashcards/ml/formula #flashcards/topic/llm Write the context window limit and explain each symbol.
?
$$n_{\text{prompt}}+n_{\text{reply}}\le n_{\text{context}}$$
- $n_{\text{prompt}}$: the tokens in everything you send (instructions, examples, pasted documents, earlier conversation).
- $n_{\text{reply}}$: the tokens the model writes back.
- $n_{\text{context}}$: the model's context window, a fixed limit.
- Every example or document you add leaves less room for the reply.

#flashcards/ml/cloze #flashcards/topic/llm Majority voting helps only if a single answer is right more than ==50%== of the time.

#flashcards/ml/example #flashcards/topic/llm With $p=0.7$, what is the accuracy of a majority of 1, 3 and 5 answers, and what happens if $p=0.4$ with $n=3$?::1 answer: 0.70. n = 3: 0.441 + 0.343 = 0.784. n = 5: 0.837. With p = 0.4 and n = 3 the result is 0.352, lower than 0.4. Voting amplifies a bias in either direction.

#flashcards/ml/example #flashcards/topic/llm The window is 8000 tokens and the prompt is 6500. How long can the reply be?::8000 − 6500 = 1500 tokens at most (8000 is an illustrative number).
