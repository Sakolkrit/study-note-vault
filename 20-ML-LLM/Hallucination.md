---
tags: [LLM, hallucination, reliability, generation, evaluation, safety]
status: growing
level: 4
tier: 3
---

# Hallucination

When a language model ([[LLM]]) writes something that **sounds right but is false or has no support**, and says it just as smoothly as a true thing.

## 🟢 1. Simple (age 12)
Imagine a friend who hates saying "I don't know". Ask them the name of the capital of a country they have never heard of, and they will still answer, in a calm voice, with a real-sounding city. They are not lying on purpose. Their habit is to always say something that sounds good.

**Example:** You ask a chatbot for a book by a made-up author, and it gives you a title, a year and a plot. Everything sounds real. The book does not exist.

## 🔵 2. How it works
1. A model writes one token after another ([[Autoregressive Generation]]). At every step a [[Softmax]] turns scores into probabilities that **must add up to 1**. There is no built-in "none of these" choice.
2. So even when the model has never learned the answer, some token still comes out on top, and it is written in the same fluent style as a true answer.
3. The model was trained to continue text in a likely way ([[Pre-training]]), not to check facts. A sentence can be very likely and still false.
4. A wrong token stays in the text. Every later token has to continue from it, so one slip can grow into a whole made-up story.
5. Kinds of mistakes people tell apart: **false facts** (wrong date or name), **made-up sources** (a paper or link that does not exist), and **ignoring the text you gave it** (you pasted an article and the summary says something the article does not).
6. What helps, in general: give the model the real text to answer from ([[RAG]]), tell it that "I don't know" is allowed ([[Prompt Engineering]]), ask it to cite or quote, and check the answer afterwards. None of these removes the problem completely.
7. The level-3 code shows the first three points in a tiny model.

![[chatgpt-hallucination-fake-url.png]]
*A chatbot is asked to summarise an article from a web address it was never able to open. It still writes a smooth, confident summary. Per the Commons description, the address was made up with real-sounding keywords, and the chatbot had no internet connection. I can see the question and the answer in the image, but not the connection claim. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ChatGPT_hallucination.png), public domain (uploaded 30 March 2023; credited to "ChatGPT" on Commons).*

## 🟠 3. Technical detail
- **Where the word comes from.** Researchers began using "hallucination" for text generation that is fluent but unsupported in tasks like summarising and translating, before chatbots. Ji et al. (2022) survey it across summarisation, dialogue, question answering, data-to-text and translation. Huang et al. (2023) survey it for large models and say it differs from earlier cases because the models are general and open-ended. (I read both abstracts only.)
- **Two broad types (common split, from memory in details).** The text contradicts the *input* you gave (a bad summary), or it contradicts the *world* (a false fact). The fixes differ: for the first, check against the input; for the second, you need outside knowledge.
- **Likely causes (named in surveys; not tested here).** The training text contains errors; the model has no stored fact for the question and fills the gap; training rewards fluent text, not "I don't know" ([[RLHF]] and [[Instruction Tuning]] can change how often it admits doubt, but I have no figure); long outputs let early errors snowball.
- **Detecting it.**
  - *Probability signals:* low top probability or high [[Entropy]] at the key tokens suggests the model is guessing.
  - *Agreement:* ask the same question several times with random sampling ([[Decoding Strategies]]); if the answers differ, the model probably does not know. If they agree, it may still be consistently wrong.
  - *Checking against a source:* compare each claim with documents ([[Evaluation Metrics]]).
- **Reducing it.**
  - *Retrieval:* put the right passages in the prompt ([[RAG]]). It helps for facts in the passages. It cannot fix a model that ignores them or a retriever that fetches the wrong page.
  - *Prompts:* allow "I don't know", ask for quotes from the given text, ask for step-by-step reasoning ([[Chain of Thought]]), and lower the randomness. These are common practice; I have **no measured effect size** for any of them in this note.
  - *Training:* fine-tune on data that includes refusals and on checked facts ([[Fine-tuning]]).
  - *Abstaining:* answer only when the model is confident enough. This works only if the probabilities are honest ([[Calibration]]).
- **The code below is a toy.** Two hand-set sets of scores stand in for a model that learned one country and not another. It shows that probabilities always add to 1, that a "best" answer always exists, and that entropy and answer-agreement tell the two cases apart.

```python
import numpy as np

cities = ["Paris", "Rome", "Oslo", "Lima", "Cairo"]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# Hand-set scores (a stand-in for a trained model): it learned France well, never saw Narnia.
scores = {"France": np.array([5.0, 1.0, 0.5, 0.2, 0.8]), "Narnia": np.array([0.4, 0.9, 0.5, 0.7, 0.6])}


def entropy(p):
    return float(-(p * np.log(p)).sum())


rng = np.random.RandomState(0)
for country, z in scores.items():
    p = softmax(z)
    print(f"{country:6s} probabilities {p.round(2)} sum {p.sum():.1f}  best '{cities[p.argmax()]}' {p.max():.2f}  entropy {entropy(p):.3f}")
    draws = rng.choice(len(cities), size=1000, p=p)
    print(f"       1000 sampled answers: same answer as the most common one {np.bincount(draws).max() / 1000:.2f} of the time")

tau = 0.5                                       # answer only if the top probability is at least tau
for country, z in scores.items():
    p = softmax(z)
    print(country, "->", cities[p.argmax()] if p.max() >= tau else "I don't know")

print("max possible entropy for 5 choices:", round(float(np.log(5)), 3))
print("entropy of (0.9, 0.05, 0.05):", round(entropy(np.array([.9, .05, .05])), 3), "| of (1/3, 1/3, 1/3):", round(entropy(np.ones(3) / 3), 3))
```

What it printed: for France the probabilities were (0.95, 0.02, 0.01, 0.01, 0.01), best answer Paris, entropy 0.266, and 1,000 sampled answers agreed with the most common one 0.94 of the time. For Narnia the probabilities were (0.16, 0.26, 0.17, 0.21, 0.19): both sets add to 1.0, and the model still has a "best" answer, Rome, at only 0.26. Its entropy was 1.594 (the most possible for 5 choices is 1.609), and samples agreed only 0.24 of the time. With a rule "answer only if the top probability is at least 0.5", France gets "Paris" and Narnia gets "I don't know".

## 🔴 4. Deep / formal
A common uncertainty signal for one step is the entropy of the model's next-token distribution:

$$
H(p)=-\sum_{i=1}^{K}p_i\ln p_i
$$

**Where:**
- $K$ is the number of choices (for an LLM, the vocabulary size) and $p_i$ is the probability of choice $i$. The $p_i$ add up to 1.
- $\ln$ is the natural log, so $H$ is in nats. It is 0 when one choice has probability 1, and largest ($\ln K$) when all choices are equally likely.
- **Matters:** high $H$ means "the model is spread over many options". A rule such as "answer only if $H$ is low" can then refuse to guess. Without such a rule, the softmax still produces a top token whatever $H$ is. Note that low $H$ does not mean true: a model can be sure and wrong.

**Example (tiny numbers):** three choices.
- Sure model, $p=(0.9,\,0.05,\,0.05)$: $-0.9\ln0.9-2\times0.05\ln0.05=0.095+0.300=\mathbf{0.394}$ nats.
- Guessing model, $p=(\tfrac13,\tfrac13,\tfrac13)$: $-3\times\tfrac13\ln\tfrac13=\ln3=\mathbf{1.099}$ nats.
- Meaning: the guessing model is nearly three times as uncertain, but an ordinary generation step would still write one of the three tokens in the same fluent way.

**Further notes (from the code, 5 choices, hand-set scores):**
- Both cases give a valid probability list and a best token. The difference between "knows" and "guesses" is only in the numbers: top probability 0.95 against 0.26, entropy 0.266 against 1.594, and answer agreement 0.94 against 0.24.
- A threshold of 0.5 separated the two toy cases. A good threshold for real models has to be chosen on real data, and the counterexample (sure but wrong) is not in the toy.
- **What the toy cannot show:** why real models hallucinate, how often, or whether their probabilities are honest. The scores are typed in by hand.
- **Open questions (stated in the abstracts):** how to measure hallucination for open-ended tasks, how far retrieval helps, and how a model can know the limits of its own knowledge (Huang et al. list this last one as a future direction).

**What came before, and how this improved on it:**
- **Before:** task-specific text generators for summarising, translating and dialogue. Their unsupported outputs were already studied under the name hallucination (Ji et al., arXiv 2202.03629, submitted 8 Feb 2022; abstract read).
- **Its limit:** such systems were built for one task, so one could compare output with the input document or reference. Large language models are general and open-ended, so hallucinations are harder to define and to check (Huang et al., arXiv 2311.05232, submitted 9 Nov 2023; abstract read).
- **What changed:** methods split into detecting (probability signals, agreement across samples, checking against sources) and reducing (retrieval, better prompts, training on checked data, abstaining). Both surveys cover this; I read only their abstracts, so which methods work best is not stated here.
- **How much better:** I have no published figure from a source I read. In this note's toy code, an "answer only if the top probability is at least 0.5" rule answered the known case and refused the unknown one. That shows the idea, not an improvement of any real model.

## Related
- [[LLM]] · [[Autoregressive Generation]] · [[Softmax]] · [[Entropy]] · [[Calibration]] · [[Decoding Strategies]] · [[RAG]] · [[Prompt Engineering]] · [[Chain of Thought]] · [[RLHF]] · [[Instruction Tuning]] · [[Fine-tuning]] · [[Evaluation Metrics]] · [[Pre-training]]

## Sources
- Ji et al., *Survey of Hallucination in Natural Language Generation*, arXiv 2202.03629 (submitted 8 Feb 2022). **Abstract only.**
- Huang et al., *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions*, arXiv 2311.05232 (submitted 9 Nov 2023). **Abstract only.**
- The advice on prompts (allow "I don't know", ask for quotes) and the split into "contradicts the input" and "contradicts the world" are common descriptions **from memory**, not checked in a paper here.
- Image: [File:ChatGPT hallucination.png](https://commons.wikimedia.org/wiki/File:ChatGPT_hallucination.png) on Wikimedia Commons, public domain per its file page. I viewed it before captioning.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Hallucination:::A fluent, confident LLM output that is false or made up, such as a wrong date, a made-up source, or an invented plot.

#flashcards/ml/concept #flashcards/topic/llm Why does a model write an answer even when it never learned it?::The softmax makes probabilities add to 1 and has no "none of these" choice, so some token always comes out on top, in the same fluent style as a true answer.

#flashcards/ml/concept #flashcards/topic/llm Why can a likely sentence still be false?::The model was trained to continue text in a likely way, not to check facts.

#flashcards/ml/concept #flashcards/topic/llm Why can one slip grow into a whole made-up story?::A wrong token stays in the text, and every later token has to continue from it.

#flashcards/ml/formula #flashcards/topic/llm Write the entropy of the next-token distribution and explain each symbol.
?
$$H(p)=-\sum_{i=1}^{K}p_i\ln p_i$$
- $K$: the number of choices (the vocabulary size for an LLM). $p_i$: the probability of choice $i$. They add to 1.
- $\ln$ is the natural log, so $H$ is in nats. It is 0 when one choice has probability 1, and largest ($\ln K$) when all are equal.
- High $H$ means the model is spread over many options, so a rule such as "answer only if $H$ is low" can refuse to guess. Without such a rule the softmax still produces a top token.
- Low $H$ does not mean true: a model can be sure and wrong.

#flashcards/ml/example #flashcards/topic/llm Three choices: $p=(0.9,0.05,0.05)$ against $(\tfrac13,\tfrac13,\tfrac13)$. What is $H$ for each?::Sure model: $0.095+0.300=0.394$ nats. Guessing model: $\ln3=1.099$ nats, nearly three times as uncertain. An ordinary generation step would still write one of the tokens fluently.

#flashcards/ml/image #flashcards/topic/llm ![[chatgpt-hallucination-fake-url.png]]
What does this picture show?
?
A chatbot asked to summarise an article from a web address it could never open still writes a smooth, confident summary. Per the Commons description the address was made up with real-sounding keywords (the note could not see the claim that the chatbot had no internet connection). (From the note's caption, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/llm A model can be sure and ==wrong==: low entropy does not mean the answer is ==true==.
