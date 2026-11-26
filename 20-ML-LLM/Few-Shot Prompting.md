---
tags: [few-shot, prompting, in-context-learning, LLM, evaluation, prompt-engineering, benchmarks]
status: growing
level: 4
tier: 2
---

# Few-Shot Prompting

**Few-shot prompting** means writing a few solved examples into the prompt, before the real question, so the model sees what kind of answer is wanted. With no example it is **zero-shot**, with one it is **one-shot**. The model's weights do not change: it only reads the examples as part of the text. Why this works at all is explained in [[In-Context Learning]]; this note is about the practical side, how many examples, which ones, and how they change a benchmark score.

## 🟢 1. Simple (age 12)
Before a quiz, a teacher shows you two questions with the answers already written in. You do not learn new facts from them, but you learn what the answer should look like: a short word, a number at the end, a full sentence. Then you answer the real question in the same style. A few-shot prompt does exactly this for an AI.

**Example:** you want the AI to answer in one short phrase. If you only ask the question, it may write a paragraph. If you first show "Question: Which factor will most likely cause a person to develop a fever? Answer: a bacterial population in the bloodstream", it tends to answer in a short phrase too.

## 🔵 2. How it works
1. **Pick the examples.** They are solved questions of the same kind as the one you care about. In a benchmark they usually come from the training split, never from the test questions.
2. **Write each example in exactly the format of the real question.** Same labels, same spacing, the answer filled in. The example shows the pattern.
3. **Put the real question last,** in the same format, with the answer left blank.
4. **Let the model continue the text** ([[LLM Inference]]), or score the possible answers ([[Log-Likelihood]]). It follows the pattern it has just seen.
5. **Read the answer.** Nothing was trained; the examples cost only extra text in the prompt ([[Context Window]]), and that text must be read again for every question.
6. **A score depends on all of these choices:** how many examples, which ones, in what order, and in what format ([[Prompt Engineering]]). Two programs that use "5-shot" can still show different examples.

No image: I found no free picture that shows this better than the prompt text itself, which is below.

## 🟠 3. Technical detail
All numbers come from the scripts below (Python 3.9.6, PyTorch 2.8.0, transformers 4.46.3, CPU; `Qwen/Qwen2.5-0.5B-Instruct`; the first 100 questions of ARC-Easy's test split). The examples are always the first k questions of the training split, written as plain text, with no chat template.

**What the one-shot prompt looks like** (the text the model reads for the first test question, as printed by the script):

```python
from datasets import load_dataset
def block(d, with_answer):
    text = "Question: " + d["question"] + "\nAnswer:"
    if with_answer:
        text += " " + d["choices"]["text"][d["choices"]["label"].index(d["answerKey"])] + "\n\n"
    return text
train = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
test = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
p = "".join(block(train[i], True) for i in range(1)) + block(test[0], False)
print(p); print(len(p), "characters")
print([len("".join(block(train[i], True) for i in range(k)) + block(test[0], False)) for k in (0, 1, 5)])
```

I saw:
```
Question: Which factor will most likely cause a person to develop a fever?
Answer: a bacterial population in the bloodstream

Question: Which statement best explains why photosynthesis is the foundation of most food webs?
Answer:
229 characters
[103, 229, 712]
```
The prompt for the first test question is 103 characters with no examples, 229 with one and 712 with five, so five examples make it 6.9 times longer (my arithmetic).

**The experiment: the same questions with 0, 1 and 5 examples.**

```python
"""ARC-Easy with 0, 1 and 5 solved examples in front of each question (first 100 test questions, Qwen2.5-0.5B-Instruct)."""
import json, math, sys, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()


@torch.no_grad()
def loglik(context, conts):
    n_ctx = len(tok(context).input_ids)
    seqs = [tok(context + c).input_ids for c in conts]
    width = max(map(len, seqs))
    ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs])
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
    return [sum(logp[i, p - 1, s[p]].item() for p in range(n_ctx, len(s))) for i, s in enumerate(seqs)]


def block(d, with_answer):
    text = "Question: " + d["question"] + "\nAnswer:"
    if with_answer:
        text += " " + d["choices"]["text"][d["choices"]["label"].index(d["answerKey"])] + "\n\n"
    return text


train = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
test = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(100))
out = {}
for k in (0, 1, 5):
    shots = "".join(block(train[i], True) for i in range(k))   # always the first k training examples
    right = 0
    for d in test:
        conts = [" " + t for t in d["choices"]["text"]]
        ll = loglik(shots + block(d, False), conts)
        right += ll.index(max(ll)) == d["choices"]["label"].index(d["answerKey"])
    out[k] = right / len(test)
    p = out[k]
    print("shots", k, "acc", p, "se", round(math.sqrt(p * (1 - p) / (len(test) - 1)), 4), "prompt chars for question 1:",
          len(shots + block(test[0], False)), flush=True)
print(json.dumps(out))
```

I saw:

| Examples shown | acc (raw log-likelihood score) | ± standard error | Prompt length for question 1 (characters) |
|---|---|---|---|
| 0 | 0.58 | 0.050 | 103 |
| 1 | 0.62 | 0.049 | 229 |
| 5 | 0.65 | 0.048 | 712 |

- **Examples shown:** how many solved training questions precede the real question.
- **acc:** the share of the 100 questions where the option with the highest log-likelihood was correct. Chance is 0.25.
- **± standard error:** how far the score could move from the luck of the questions (see [[LLM Evaluation Lab]]).
- **Prompt length:** the cost side. Every question carries the examples again.

**Summary: what the table shows.** More examples gave a higher score, 0.58, 0.62 and 0.65, and every extra example makes the prompt longer. Caution: the largest gap is 7 points (0.65 against 0.58) and each score has an error of about 5 points, so I cannot say the gain is real. I did not save the per-question results, so I could not run the paired test that the evaluation lab uses. Only one model, 100 questions, the first k training examples (no other choice or order tried).

**An example of two "5-shot" prompts that differ** (from [[LLM Evaluation Lab]], GSM8K with the 0.5B model). My evaluator always showed the first five training problems, the first being Natalia's clips. The official lm-evaluation-harness showed five other problems (its prompt for the first test question started with a problem about Jen and Tyler and was 3,187 characters long). With the same model and questions, the strict scores agreed on only 24 of 30 verdicts (4 right in both programs), and the flexible scores were 8 and 7 of 30 right. The examples also set the answer format: strict scoring needs the final line `#### number`, which the examples demonstrate.

## 🔴 4. Deep / formal
There is no formula to learn here: a few-shot prompt is only text. What deserves care is the design.

**Choices that change the result** (general knowledge unless marked; I tested only the number of examples):
- **How many examples:** more usually helps up to a point, then the prompt gets long and the gain flattens. The GPT-3 paper used K from 10 to 100, "what fits" in its 2,048-token window (as noted in [[GPT-3 2020]]).
- **Which examples:** examples close to the question usually help more than random ones.
- **Their order:** the same examples in another order can change the answer.
- **Label balance:** if most examples have the same answer, the model leans towards it.
- **The format:** the model copies the layout it sees, including the answer format ("#### number"), which is why the strict score needs it.
- **The model's training:** an instruction-tuned model ([[Instruction Tuning]]) may need fewer examples, and may behave differently with its chat template. I used raw text and did not test the template.
- **Cost:** the examples are part of every prompt, so five examples made the first prompt 6.9 times longer (my arithmetic). Scoring four options re-reads that prefix four times unless the program caches it ([[Prompt Caching]]).

**Few-shot against chain of thought.** For maths, the examples can also show the working, not only the answer ([[Chain of Thought]]). The GSM8K training answers my evaluator used contain the step-by-step solution before the `####` line.

**Limits and trade-offs:**
- A benchmark's score then depends on the prompt as well as the model, so two published numbers with different shots or examples are not directly comparable (my lab saw two 5-shot prompts differ).
- It cannot teach a model what it cannot do. It mostly shows the format and the task.
- Test questions must not be among the examples, and if they appeared in the training text the score may measure memory ([[Data Contamination]]).

**What came before, and how this improved on it:**
- **Before:** pretrain a model, then fine-tune a separate copy for every task, which the GPT-3 abstract says needs task-specific datasets of thousands or tens of thousands of examples (as noted in [[GPT-3 2020]]; [[Fine-tuning]]).
- **Its limit:** a labelled dataset and a training run for every new task.
- **What changed:** GPT-3 (Brown and others, 2020) showed that a large enough model can be given the task as text and a few examples, with no gradient update. The paper compares zero-, one- and few-shot settings.
- **How much better:** published, from my note of the paper: LAMBADA accuracy 76.2% zero-shot against 86.4% few-shot, 10.2 points more (my arithmetic). In my own run, 0.58 with no examples and 0.65 with five on 100 ARC-Easy questions with a 0.5B model, a gap within the noise.

## Related
[[In-Context Learning]] · [[Prompt Engineering]] · [[Chain of Thought]] · [[Instruction Tuning]] · [[Fine-tuning]] · [[Context Window]] · [[Prompt Caching]] · [[Log-Likelihood]] · [[Benchmarks]] · [[LLM Evaluation Lab]] · [[GPT-3 2020]] · [[Data Contamination]] · [[LLM]]

## Sources
- My own runs (Python 3.9.6, PyTorch 2.8.0, transformers 4.46.3, CPU; Qwen2.5-0.5B-Instruct, Apache-2.0; ARC-Easy from `allenai/ai2_arc`, CC BY-SA 4.0 per its dataset card).
- GPT-3 paper, as read in [[GPT-3 2020]] (abstract, section 2 and the LAMBADA table). The GSM8K prompt comparison is from [[LLM Evaluation Lab]].
- Marked "general knowledge": the list of design choices. Not checked: the note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Few-shot prompting:::Writing a few solved examples into the prompt before the real question, so the model sees the task and the answer format. No weights change. Zero-shot has no example, one-shot has one.

#flashcards/ml/concept #flashcards/topic/llm What are the steps of building a few-shot prompt?::Pick solved examples of the same kind (from the training split), write each in exactly the format of the real question, put the real question last with the answer blank, then let the model continue or score the options.

#flashcards/ml/concept #flashcards/topic/llm What did 0, 1 and 5 examples do on ARC-Easy for the 0.5B model?::Accuracy 0.58, 0.62 and 0.65 on the first 100 questions, each with an error of about 0.05. The 7-point gap is within the noise, and only the first k training examples were tried (my run).

#flashcards/ml/example #flashcards/topic/llm How much longer did 5 examples make the prompt for the first ARC-Easy question?::From 103 to 712 characters, 6.9 times longer (my arithmetic). Every question carries the examples again, so this is paid on every call.

#flashcards/ml/concept #flashcards/topic/llm Why can two "5-shot" scores of the same model differ?::The five examples can differ. My evaluator showed the first five GSM8K training problems and the official harness showed five others (a 3,187-character prompt). The strict verdicts agreed on 24 of 30 questions, and the flexible scores were 8 and 7 of 30.

#flashcards/ml/concept #flashcards/topic/llm Why do few-shot examples affect the strict GSM8K score?::They show the answer format. Strict scoring needs the final line #### number, which the examples demonstrate. A model that writes the right number without that line is missed by strict and counted by flexible.

#flashcards/ml/concept #flashcards/topic/llm Which choices can change a few-shot result?::How many examples, which ones, their order, the balance of their answers, their format, and whether a chat template is used. I tested only the number of examples; the rest is general knowledge.

#flashcards/ml/concept #flashcards/topic/llm What did few-shot prompting replace, according to the GPT-3 paper?::Fine-tuning a separate model for every task, which needs thousands or tens of thousands of labelled examples per task (abstract, as noted in my GPT-3 note). GPT-3 gets the task as text and a few examples, with no gradient update.

#flashcards/ml/concept #flashcards/topic/llm What is the published gain from few-shot on LAMBADA in GPT-3?::76.2% zero-shot to 86.4% few-shot, 10.2 points more (from my note of the paper, my arithmetic).

#flashcards/ml/concept #flashcards/topic/llm What does few-shot prompting not do?::It does not train the model and cannot teach what it cannot do; it mostly shows the task and the format. It also adds cost to every prompt, and a leaked test set can make a high score mean memory.

#flashcards/ml/cloze #flashcards/topic/llm In few-shot prompting the examples go ==before== the real question, and the model's ==weights== are not changed.
