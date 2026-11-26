---
tags: [log-likelihood, probability, evaluation, LLM, language-model, benchmarks, ML]
status: growing
level: 4
tier: 2
---

# Log-Likelihood

The **log-likelihood** of some text is the logarithm of the probability a model gives to that text. For a language model it is the sum, over the tokens, of the log of the probability of each token given the tokens before it. It is a score for "how unsurprising is this text to the model". Training a language model pushes it up, and multiple-choice benchmarks use it to pick the model's answer.

## 🟢 1. Simple (age 12)
Imagine a friend who tries to guess every next word of a sentence and says how sure they are each time. If they were very sure about every word and right, the sentence was easy for them. If they were surprised again and again, it was hard. The log-likelihood is a single number that adds up all that surprise: closer to zero means "not surprised", a big negative number means "very surprised".

**Example:** you ask a friend "What do plants need to grow?" and offer two answers, "sunlight and water" or "moon dust". Your friend finds the first much less surprising. If a test says "pick the answer your friend finds less surprising", the friend picks "sunlight and water" without ever saying a letter.

## 🔵 2. How it works
1. A [[LLM]] reads text one [[Tokenization|token]] at a time. After each token it gives a probability to every possible next token (see [[Softmax]]).
2. For a piece of text we look up the probability the model gave to the token that really came next, at each position. Each is a number between 0 and 1.
3. The probability of the whole text is these numbers multiplied together. But 100 numbers like 0.1 multiplied together become so tiny that a computer rounds them to zero.
4. So we take the **logarithm** of each probability first and **add** instead of multiply. A probability of 1 becomes 0, and smaller probabilities become bigger and bigger negative numbers. The sum is the log-likelihood.
5. **Use for a quiz:** write the question, then try each answer option as the continuation, and compute the log-likelihood of the option's tokens given the question. The option with the highest (least negative) value is the model's answer. Nobody asks the model to say "A" or "B".
6. **Use for training:** the loss that trains a language model ([[Cross-Entropy Loss]]) is the average negative log-likelihood of the real next tokens, so training is the same quantity pushed in the right direction ([[Maximum Likelihood]]).

No image: I found no free picture that shows this on a real model, and a made-up one would not help. The numbers below come from a real run instead.

## 🟠 3. Technical detail
All numbers come from the two scripts below (Python 3.9.6, PyTorch 2.8.0, transformers 4.46.3, CPU; the model is `Qwen/Qwen2.5-0.5B-Instruct`). This is the same scoring my evaluator uses in [[LLM Evaluation Lab]].

**The arithmetic on three made-up token probabilities, and why we add logs.**

```python
import math
# one answer made of three tokens, with the probability the model gave each token
p = [0.5, 0.25, 0.8]
logs = [math.log(x) for x in p]
print("token log-probs", [round(v, 3) for v in logs], "sum", round(sum(logs), 3),
      "probability of the whole answer", round(math.exp(sum(logs)), 3), "product", round(math.prod(p), 3))
# why we add logs: the product of many small numbers underflows
print("400 tokens at p=0.1: product", math.prod([0.1] * 400), "| sum of logs", round(400 * math.log(0.1), 2))
# length: one likely-looking short answer against a longer answer that is better token by token
short, long_ = [0.30], [0.5] * 4
for name, a in (("short", short), ("long", long_)):
    ll = sum(math.log(x) for x in a)
    print(name, "tokens", len(a), "log-likelihood", round(ll, 3), "per token", round(ll / len(a), 3))
```

I saw:
```
token log-probs [-0.693, -1.386, -0.223] sum -2.303 probability of the whole answer 0.1 product 0.1
400 tokens at p=0.1: product 0.0 | sum of logs -921.03
short tokens 1 log-likelihood -1.204 per token -1.204
long tokens 4 log-likelihood -2.773 per token -0.693
```
The product of 400 tokens underflowed to 0.0, while the sum of logs is a perfectly good number (-921.03). The last two lines show the length problem: the long answer is worse in total (-2.773 against -1.204) but better per token (-0.693 against -1.204).

**The same thing on a real model: one ARC-Easy question, four options, token by token.**

```python
"""Per-token log-probabilities of the four answer options of one ARC-Easy question (Qwen2.5-0.5B-Instruct)."""
import math, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
d = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")[59]
context = "Question: " + d["question"] + "\nAnswer:"
print(context)
n_ctx = len(tok(context).input_ids)
for label, text in zip(d["choices"]["label"], d["choices"]["text"]):
    ids = tok(context + " " + text, return_tensors="pt").input_ids
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
    pieces = [(tok.decode(ids[0, p]), round(logp[p - 1, ids[0, p]].item(), 3)) for p in range(n_ctx, ids.shape[1])]
    total = sum(v for _, v in pieces)
    print(label, repr(text), pieces, "sum", round(total, 3), "probability", f"{math.exp(total):.2e}",
          "per byte", round(total / len((' ' + text).encode()), 3), "gold" if label == d["answerKey"] else "")
```

The question is "Cells need nutrients for energy. Which system is responsible for breaking down food to provide cellular energy?" and I saw:

| Option | Token pieces and their log-probabilities | Sum | Probability of the option | Per byte (my rule) | Per character without the space (harness rule, my arithmetic) |
|---|---|---|---|---|---|
| A digestive (correct) | ` digestive` −10.159 | −10.159 | 3.87e-05 | −1.016 | −1.129 |
| B excretory | ` ex` −11.553, `cret` −3.172, `ory` −0.005 | −14.730 | 4.01e-07 | −1.473 | −1.637 |
| C circulatory | ` circ` −12.030, `ulatory` −0.202 | −12.232 | 4.87e-06 | −1.019 | −1.112 |
| D respiratory | ` respiratory` −14.591 | −14.591 | 4.60e-07 | −1.216 | −1.326 |

- **Sum:** the log-likelihood of the option given the question. The highest is A, the correct answer.
- **Probability of the option:** e to the power of the sum. These do not add up to 1: they are the probabilities of four particular texts, not a share among options. A is 84 times more likely than D (my arithmetic).
- **Per byte and per character:** the sum divided by the option's length, the two ways of adjusting for length that decided one question between my evaluator and the official harness. With the per-byte rule A still wins by a hair (−1.016 against −1.019). With the harness's per-character rule C wins (−1.112 against −1.129) and is wrong. One tiny rule decides this question.

**Summary: what the table shows.** The log-likelihood picks the right option here, and the length adjustment, meant to be fair, flips the answer by 0.017 per byte. Caution: this is one question on a 0.5B model. A word made of several pieces (`ex`, `cret`, `ory`) pays a large price for its first piece and almost nothing for the rest, so the total depends on how the [[Tokenization|tokenizer]] splits the word.

## 🔴 4. Deep / formal
**Log-likelihood of a continuation $y$ given a context $x$** (the chain rule of probability, then a logarithm):

$$
\log P(y \mid x) \;=\; \sum_{t=1}^{T} \log P\big(y_t \mid x,\, y_1,\dots,y_{t-1}\big)
$$

**Where:**
- $x$ is the context (for a quiz, the question and the cue "Answer:"). **Matters:** the same option scores differently after a different question, which is the whole point.
- $y_1,\dots,y_T$ are the tokens of the candidate text. **Matters:** the count $T$ depends on the tokenizer, so scores of different tokenizers are not directly comparable.
- $P(y_t \mid x, y_{<t})$ is the probability the model gives to the real token at position $t$, given everything before it. **Matters:** it is read off the model's softmax output; each factor is at most 1, so each log is at most 0.
- The sum over $t$ comes from the product rule turned into a sum by the logarithm. **Matters:** it avoids the underflow shown above and turns a product into something that can be averaged and differentiated.

**Example (tiny numbers):** a three-token answer whose tokens got probabilities 0.5, 0.25 and 0.8.
- Logs: ln 0.5 = −0.693, ln 0.25 = −1.386, ln 0.8 = −0.223.
- Sum: −0.693 − 1.386 − 0.223 = −2.303.
- Check: e to the power of −2.303 is 0.1, and 0.5 × 0.25 × 0.8 is also 0.1.
- The answer means: the model gave this exact three-token text a probability of 0.1; a rival text with a sum above −2.303 is more likely.

**Length adjustments (the part that differs between programs).**
- *Raw sum:* favours short answers, because every extra token multiplies in a number below 1.
- *Mean per token:* removes that favour but depends on how the text is cut into tokens.
- *Per byte or per character:* does not depend on the tokenizer, but the choice of what counts (the leading space, bytes against characters) can flip a close question, as shown above. The official harness divides by the number of characters without the leading space, which I inferred from its output and did not read in its code (see [[LLM Evaluation Lab]]).
- *The GPT-3 paper's other normalisation:* it divides the option's probability given the question by its probability given only a generic cue ("Answer:"), for ARC, OpenBookQA and RACE, and uses per-token likelihood for most other tasks (paper, section 2.4; my note [[GPT-3 2020]]).

**Relations to other quantities.**
- The training loss of a language model is the mean of the negative log-likelihood of the real tokens ([[Cross-Entropy Loss]], [[Maximum Likelihood]]).
- [[Perplexity]] is e to the power of that mean negative log-likelihood per token.
- The log-likelihood is a statement about the model's belief, not about truth. A fluent wrong answer can score above an awkward correct one.

**Limits and trade-offs:**
- It scores the text of an option, not its meaning: "The digestive system" and "digestive" get different scores for the same idea (general knowledge; I did not test it).
- It depends on the prompt wording and any few-shot examples ([[Few-Shot Prompting]]).
- It cannot score open answers (maths, code), where the model must write the answer and a program checks it. There the answer format starts to matter: in my lab the 0.5B model's right numbers were missed by the strict check in 4 of 8 cases ([[LLM Evaluation Lab]]).
- If test questions leaked into the training text, a high likelihood may mean memory ([[Data Contamination]]).

**What came before, and how this improved on it:**
- **Before:** ask the model to generate an answer and compare the text with the key (string matching or, later, a judge model), the usual way to test a generator.
- **Its limit:** the model must follow the answer format, and a small model often does not. In my lab, the 0.5B model wrote the right final number but not in the required `#### number` form in 4 of its 8 right answers on GSM8K (30 questions, strict 0.133 against flexible 0.267, my run).
- **What changed:** score each given option by its log-likelihood, so there is no formatting step and a score can be computed in one forward pass per option.
- **How much better:** I did not run a head-to-head on the same task. In my run, this scoring gave the 0.5B model ARC-Easy `acc` 0.565 (chance 0.25) on the first 200 questions with no format failures possible. Not compared with letter-picking or generation on that task.

## Related
[[Perplexity]] · [[Cross-Entropy Loss]] · [[Maximum Likelihood]] · [[Softmax]] · [[Tokenization]] · [[Probability]] · [[Benchmarks]] · [[Evaluation Metrics]] · [[LLM Evaluation Lab]] · [[GPT-3 2020]] · [[Few-Shot Prompting]] · [[Data Contamination]] · [[LLM]]

## Sources
- My own runs (Python 3.9.6, PyTorch 2.8.0, transformers 4.46.3, CPU; Qwen2.5-0.5B-Instruct, Apache-2.0; ARC-Easy from `allenai/ai2_arc`, CC BY-SA 4.0 per its dataset card).
- GPT-3 paper section 2.4 as read in [[GPT-3 2020]]. The harness's length rule and the generation-format numbers come from [[LLM Evaluation Lab]].
- Definitions of log-likelihood, perplexity and the chain rule are general knowledge. Not checked: the note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Log-likelihood of a text:::The logarithm of the probability a model gives to the text; for a language model, the sum over tokens of the log-probability of each token given the ones before it.

#flashcards/ml/concept #flashcards/topic/evaluation Why do we add log-probabilities instead of multiplying probabilities?::The product of many small numbers underflows to zero: 400 tokens at probability 0.1 give 0.0 in floating point, while the sum of logs is a fine number (-921.03, my run).

#flashcards/ml/concept #flashcards/topic/evaluation How does a benchmark use log-likelihood to answer a multiple-choice question?::It appends each option to the question, computes the log-likelihood of the option's tokens, and picks the option with the highest value. The model never says a letter.

#flashcards/ml/formula #flashcards/topic/evaluation Write the log-likelihood of a continuation given a context and explain each symbol.
?
$$\log P(y \mid x) = \sum_{t=1}^{T} \log P\big(y_t \mid x, y_1,\dots,y_{t-1}\big)$$
- $x$: the context, such as the question and "Answer:"; the same option scores differently after another question.
- $y_1,\dots,y_T$: the tokens of the candidate text; $T$ depends on the tokenizer.
- $P(y_t \mid x, y_{<t})$: the probability the model gives the real token at position $t$; at most 1, so each log is at most 0.
- The sum: the product rule turned into a sum by the logarithm; it avoids underflow.

#flashcards/ml/example #flashcards/topic/evaluation Token probabilities 0.5, 0.25 and 0.8: what is the log-likelihood?::-0.693 - 1.386 - 0.223 = -2.303. Check: e to the power of -2.303 is 0.1, and 0.5 x 0.25 x 0.8 is also 0.1, so the whole answer has probability 0.1 (my arithmetic).

#flashcards/ml/concept #flashcards/topic/evaluation Why is the raw log-likelihood unfair to long answers, and what is the fix?::Every extra token multiplies in a number below 1, so a longer answer scores lower in total. Divide by the length (per token, byte or character). Toy: one token at 0.30 gives -1.204; four tokens at 0.5 give -2.773 in total but -0.693 per token.

#flashcards/ml/concept #flashcards/topic/evaluation On ARC-Easy question 59, which option wins by raw sum and which by the harness's length rule?::By raw sum and by per byte, "digestive" (-10.159) wins and is correct. With the harness's per-character rule (without the leading space), "circulatory" wins (-1.112 against -1.129) and is wrong. My run, 0.5B model.

#flashcards/ml/concept #flashcards/topic/evaluation Why does the tokenizer matter for log-likelihood scores?::The sum has one term per token, and a word split into several pieces pays a big price for its first piece. In my run "excretory" was ` ex` (-11.553), `cret` (-3.172), `ory` (-0.005). Scores from different tokenizers are not directly comparable.

#flashcards/ml/concept #flashcards/topic/evaluation Do the option probabilities from log-likelihood scoring add up to 1?::No. They are the probabilities of four particular texts, not a share among options. In my run they were 3.87e-05, 4.01e-07, 4.87e-06 and 4.60e-07.

#flashcards/ml/concept #flashcards/topic/evaluation How is log-likelihood related to the training loss and to perplexity?::The loss is the mean negative log-likelihood of the real next tokens, and perplexity is e to the power of that mean per token (general knowledge).

#flashcards/ml/concept #flashcards/topic/evaluation What can log-likelihood scoring not do?::Score open answers such as maths or code, where the model must write the answer and a program checks it. It also scores the wording of an option, not its meaning, and a leaked test set can make a high score mean memory.

#flashcards/ml/cloze #flashcards/topic/evaluation The log-likelihood of a text is the ==sum== of the ==log-probabilities== of its tokens, each given the tokens before it.
