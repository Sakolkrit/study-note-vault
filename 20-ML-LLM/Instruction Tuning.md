---
tags: [LLM, fine-tuning, instruction-tuning, SFT, alignment, training]
status: growing
level: 4
tier: 3
---

# Instruction Tuning

Also called **supervised fine-tuning (SFT)** on instructions. Teaching a pretrained model to **follow requests**, by training it on examples of a request and a good reply.

## 🟢 1. Simple (age 12)
A model that has only read huge piles of text is like a person who has read every book but never had a conversation. Ask it "can I put my teddy bear in the washer?" and it may just **keep writing more of the question's text** instead of answering. Instruction tuning is **practice conversations**: thousands of examples of "here is a request, here is a good answer", until the model learns to behave like a helpful assistant.

**Example:** Before: "Can I put my teddy bear in the washer?" → "Teddy bears are often made of materials like polyester and cotton, with plastic eyes..." (it carries on describing). After: "No, it might get damaged. Try hand washing instead."

## 🔵 2. How it works
1. First the model is **pretrained** on a vast amount of text to predict the next token ([[Pre-training]]). It learns language and facts, but not to take requests.
2. Collect **pairs**: an instruction (write a story, make a poem, list three activities, explain why) and the reply you would like (the lecture's examples).
3. Train the model with the **same next-token objective**, but on these pairs: given the request, predict the reply, token by token ([[Fine-tuning]]).
4. The pairs can be written by people, written by other models (synthetic), or a mix. They cover assistant chats, maths, reasoning, code and safety.
5. The lecture gives the size as roughly **thousands to millions of examples**, far less than pretraining.
6. The result is a model that answers requests directly. Tricks like [[LoRA]] make this cheaper.
7. It is not the end: **preference tuning** ([[RLHF]], [[Direct Preference Optimization]]) comes next to make the model behave better, for example refusing harmful requests.

## 🟠 3. Technical detail
- **Same loss, different data:** it is ordinary next-token training. The data is formatted as a prompt followed by a response.
- **Loss on the reply only:** in practice the loss is usually computed on the response tokens, with the prompt masked (common practice; the lecture states the objective as "predict next token given the input" and I did not check how it treats the prompt). The code shows that masked loss equals the average over response positions only.
- **Data quality dominates:** the lecture lists the downsides: it needs very high-quality data, it is sensitive to the distribution of prompts, generalisation is uncertain, evaluation is hard, and it is computationally expensive.
- **Research:** the FLAN paper instruction-tuned a 137-billion-parameter model on over 60 language tasks written as instructions, and found it improved zero-shot performance on unseen task types, beating zero-shot GPT-3 175B on 20 of 25 tasks. Its ablations credit the number of datasets, model scale and natural-language instructions (arXiv 2109.01652 abstract, read).
- **Evaluation:** benchmarks for knowledge, reasoning, maths and code (see [[Benchmarks]]), plus human-vote sites. The lecture warns that training on a test task can confuse comparisons.
- **Where it sits in the life of a model:** pretraining, then instruction tuning, then preference tuning (the lecture's "lifecycle").
- **Parameter-efficient version:** train a small add-on rather than all weights ([[LoRA]]).

```python
import torch
import torch.nn.functional as F
torch.manual_seed(0)
# one instruction-tuning example, as the model sees it: prompt tokens then response tokens
prompt = 'Write a poem about my teddy bear .'.split()
response = 'Soft and cuddly , full of charm'.split()
vocab = {w: i for i, w in enumerate(sorted(set(prompt + response)))}
ids = torch.tensor([vocab[w] for w in prompt + response])
V = len(vocab)
logits = torch.randn(len(ids), V)                              # stand-in for the model's outputs at each position
# next-token training: position i must predict token i+1
x_logits, target = logits[:-1], ids[1:].clone()
full = F.cross_entropy(x_logits, target)                       # loss over every next-token prediction, prompt included
labels = target.clone(); labels[: len(prompt) - 1] = -100      # hide the targets that are still inside the prompt
masked = F.cross_entropy(x_logits, labels, ignore_index=-100)  # loss over the response tokens only
per = F.cross_entropy(x_logits, target, reduction='none')
print('tokens:', len(ids), '| prompt', len(prompt), '| response', len(response), '| predictions scored with masking:', int((labels != -100).sum()), 'of', len(target))
print('loss over all tokens', round(full.item(), 4), '| loss over the response only', round(masked.item(), 4))
print('response-only loss by hand:', round(per[len(prompt) - 1:].mean().item(), 4), '| matches:', abs(per[len(prompt) - 1:].mean().item() - masked.item()) < 1e-6)
# the data: input and desired output pairs, not just text
pairs = [('Write a short story about a teddy bear who likes poetry.', 'Once upon a time, a bear, Teddy, found some verses.'), ('List three activities for a rainy day.', 'Sure! 1. Read. 2. Hug. 3. Nap.')]
for q, a in pairs: print('example:', len(q.split()), 'prompt words ->', len(a.split()), 'response words')
```

## 🔴 4. Deep / formal
**The loss.** For an instruction $x$ with a reply $y=(y_1,\dots,y_m)$:

$$
\mathcal{L}_{\text{SFT}}=-\frac{1}{m}\sum_{t=1}^{m}\log p_\theta\big(y_t\mid x,\,y_{<t}\big)
$$

**Where:**
- $x$ is the instruction (the prompt) and $y_t$ is the $t$-th token of the desired reply, with $y_{<t}$ the reply tokens before it.
- $p_\theta(\cdot)$ is the probability the model (with weights $\theta$) gives to the right next token.
- $m$ is the number of reply tokens, and $-\log p$ is the cost of a wrong or unsure guess (this is the cross-entropy, see [[Loss Function]]).
- **Matters:** training lowers this number, so the model learns to produce the reply when given the instruction. The sum is only over reply tokens in this version (the prompt tokens are not scored). Without the mask, the model would also be trained to *predict the user's own words*, which wastes effort.

**Example (tiny numbers):** a reply of 3 tokens that the model gives probabilities $0.5$, $0.25$ and $0.125$.
- Costs: $-\ln0.5=\mathbf{0.693}$, $-\ln0.25=\mathbf{1.386}$, $-\ln0.125=\mathbf{2.079}$.
- Average: $(0.693+1.386+2.079)/3=\mathbf{1.386}$.
- Meaning: the model is fairly unsure. After training, those probabilities rise and the loss falls toward 0.

**Further notes:**
- **In the code:** a 15-token example (8 prompt tokens, 7 reply tokens) has 14 next-token predictions. Masking leaves 7 of them. Random stand-in scores gave a loss of $3.1344$ over all and $3.0705$ over the reply only; the by-hand reply-only average matched. No real model is trained.
- **Why "instruction tuning" is a special case of SFT:** SFT can use any input and output pairs (for example, spam detection). Instruction tuning uses instructions across many tasks.
- **Not covered:** chat templates, how many examples are enough, multi-turn data, and synthetic data pitfalls.

**What came before, and how this improved on it:**
- **Before:** a pretrained model used through prompting alone, with examples written into the prompt ([[In-Context Learning]]), as in GPT-3 (named in the lecture).
- **Its limit:** the lecture shows a pretrained model continuing the question's text instead of answering it, and prompting costs effort, tokens and delay each time.
- **What changed:** fine-tune on many tasks written as instructions with desired replies (Wei et al., FLAN, arXiv 2109.01652, 3 September 2021, abstract read).
- **How much better:** the FLAN abstract reports surpassing zero-shot GPT-3 175B on 20 of 25 tasks with a 137B model, and beating few-shot GPT-3 by a large margin on several tasks. I did not reproduce this. In this note's code, only the arithmetic of the loss is checked.

## Related
- [[Fine-tuning]] · [[Pre-training]] · [[LoRA]] · [[RLHF]] · [[Direct Preference Optimization]] · [[In-Context Learning]] · [[Benchmarks]] · [[LLM]] · [[Loss Function]] · [[Prompt Engineering]] · [[Knowledge Distillation]] · [[InstructGPT 2022]] · [[Chat Template]] · [[System Prompt]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 4 slides, Afshine and Shervine Amidi
- Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN), 2021 (arXiv 2109.01652, abstract read)
- Dominguez-Olmedo et al., *Training on the Test Task Confounds Evaluation and Emergence*, 2024 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Instruction tuning:::Practice conversations. Train a pretrained model on pairs of an instruction and a good reply until it behaves like a helpful assistant instead of just continuing the text.

#flashcards/ml/formula #flashcards/topic/llm Write the instruction-tuning (SFT) loss and explain each symbol.
?
$$\mathcal{L}_{\text{SFT}}=-\frac{1}{m}\sum_{t=1}^{m}\log p_\theta\big(y_t\mid x,\,y_{<t}\big)$$
- $x$: the instruction (the prompt). $y_t$: the $t$-th token of the desired reply. $y_{<t}$: the reply tokens before it.
- $p_\theta(\cdot)$: the probability the model (weights $\theta$) gives to the right next token.
- $m$: the number of reply tokens. $-\log p$: the cost of a wrong or unsure guess (cross-entropy).
- Training lowers this number so the model produces the reply when given the instruction. In this version the sum is over reply tokens only (the prompt tokens are not scored).

#flashcards/ml/example #flashcards/topic/llm A 3-token reply gets probabilities 0.5, 0.25, 0.125. What is the loss?::Costs $0.693$, $1.386$, $2.079$. Average $1.386$. The model is fairly unsure. After training those probabilities rise and the loss falls toward 0.

#flashcards/ml/cloze #flashcards/topic/llm A model that only read huge piles of text may ==keep writing== the question instead of answering it. Instruction tuning fixes that with ==request and answer== pairs.

#flashcards/ml/concept #flashcards/topic/llm What comes before instruction tuning?::Pre-training on a vast amount of text to predict the next token. It learns language and facts but not to take requests.
