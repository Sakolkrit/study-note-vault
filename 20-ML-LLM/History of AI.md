---
tags: [history, AI, timeline, neural-network, LLM, overview]
status: growing
level: 4
tier: 3
---

# History of AI

The story of how people tried to build thinking machines, told as a chain: each idea fixed a weakness of the one before it.

## 🟢 1. Simple (age 12)
People first tried to **write down the rules** of thinking and give them to a computer. That worked for narrow games and puzzles, but not for seeing or talking. Then people tried **letting the computer learn from examples** instead. Small learners were weak, so over decades they grew bigger, got more data and faster chips. Today's chat assistants are the latest step in that chain.

**Example:** Teaching a computer to spot a cat. Rule-writing: "has pointy ears, whiskers..." (you quickly run out of rules). Learning: show a million cat photos and let the computer find the patterns itself.

## 🔵 2. How it works
The story has three big stages, and in each the next stage fixed a limit of the one before.
1. **Rules and logic (1950s to 1980s).** Humans write the knowledge down by hand. Good for narrow, clear problems. Fails when the world is messy.
2. **Learning from data (1980s to 2000s).** A [[Machine Learning]] model finds patterns in examples ([[Supervised Learning]]). Humans still design the features.
3. **Deep learning (2010s on).** A [[Neural Network]] with many layers learns the features too, trained by [[Backpropagation]] on [[GPU]]s. It beat hand-made features first in pictures ([[History of Image Classification]]), then in language ([[History of Text Analytics]]).
4. **Scale and language models (2017 on).** The [[Transformer]] made training parallel, and [[Pre-training]] on huge text produced [[LLM]]s. [[RLHF]] and then [[Reasoning Model]]s shaped what they do.
5. Two other threads run through all of this: the **sequence thread** ([[From FNN to RNN]]) and the **learning-by-trying thread** ([[Reinforcement Learning]]).

## 🟠 3. Technical detail
A timeline. "Checked" means the date was confirmed this session; "memory" means I wrote it from memory and did not check it.

| Year | Event | What it fixed | Source status |
|---|---|---|---|
| 1943 | McCulloch and Pitts: first mathematical model of a neural network | gave the idea of a computing neuron | checked (Wikipedia timeline page) |
| 1950 | Turing, *Computing Machinery and Intelligence*, proposes the Turing test | a way to ask the question | checked |
| 1956 | Dartmouth summer workshop, where McCarthy coined "artificial intelligence" | named the field | checked |
| 1958 | Rosenblatt's [[Perceptron]], a neuron that learns | learning instead of hand-set weights | memory |
| 1969 | Minsky and Papert, *Perceptrons* | showed single neurons cannot do XOR; some say it started the 1970s "AI winter" | checked (book date, "some say" winter) |
| 1973 | Lighthill report, a negative verdict on AI research | funding cuts | checked (listed on the page) |
| mid-1980s | [[Backpropagation]] widely used to train multi-layer networks | train hidden layers | checked ("widely used mid-1980s"); Rumelhart, Hinton and Williams 1986 from memory |
| 1989 to 1998 | LeNet reads zip codes and checks | first useful convolutional network | checked in [[LeNet and AlexNet]] (Wikipedia read) |
| 1997 | LSTM (Hochreiter and Schmidhuber); Deep Blue beats Kasparov | memory over long sequences; machine chess champion | checked |
| 2012 | AlexNet wins ImageNet with about half the errors of the runner-up | deep learning on GPUs beats hand-made features | checked |
| 2013 | word2vec from Google researchers | cheap, good word vectors | checked |
| 2014 | GAN (Goodfellow et al.); sequence-to-sequence; attention | generating, translating, looking back | checked on arXiv |
| 2015 | ResNet, 152 layers | very deep networks that train | checked on arXiv |
| 2017 | Transformer | parallel training, no recurrence | checked on arXiv |
| 2018 | BERT | pre-train then fine-tune | checked on arXiv |
| 2020 | GPT-3 with 175B parameters | few-shot prompting | checked on arXiv |
| 2022 | InstructGPT and ChatGPT, trained with human feedback ([[RLHF]]) | made models follow instructions | memory (dates) |
| 2023 | LLaMA: open models, 13B beating GPT-3 175B on most benchmarks | open, efficient models | checked on arXiv |
| 2025 | DeepSeek-R1: reasoning from pure reinforcement learning | reasoning without human-written steps | checked on arXiv (abstract) |

Papers on this chain have their own notes in `70-Research`: [[AlexNet 2012]], [[VGG 2014]], [[ResNet 2015]], [[R-CNN 2013]], [[YOLO 2015]], [[ViT 2020]], [[Seq2Seq 2014]], [[Word2Vec 2013]], [[Attention Is All You Need 2017]], [[BERT 2018]], [[RoFormer 2021]], [[GPT-3 2020]], [[LLaMA 2023]], [[DeepSeek-R1 2025]].

The code turns published figures from the checked rows into ratios.

```python
# Published ImageNet top-5 error rates quoted in this vault's notes, and what they mean in relative terms.
alex, resnet = 15.3, 3.57          # 2012 winner (Wikipedia, read), 2015 ensemble (ResNet abstract, read)
print("AlexNet 2012 top-5 error:", alex, "%  ->  ResNet 2015 ensemble:", resnet, "%")
print("relative drop in errors:", round(100 * (alex - resnet) / alex, 1), "%   (", round(alex / resnet, 2), "x fewer mistakes )")
# parameter counts the vault computed itself (see LeNet and AlexNet, ResNet)
print("AlexNet layers 8, parameters 61,100,840;  ResNet-152 layers 152, parameters 60,192,808")
print("layers x", round(152 / 8, 1), " parameters ratio", round(60_192_808 / 61_100_840, 3))
# R-CNN vs Fast R-CNN (abstract): training 9x faster, testing 213x faster
print("Fast R-CNN over R-CNN: training 9x faster, test-time 213x faster (abstract); YOLO 45 frames/s (abstract) -> ms per image:", round(1000 / 45, 1))

# Published numbers (abstracts read) and what they mean as ratios.
print("Seq2Seq LSTM 34.8 BLEU vs phrase-based SMT 33.3 on WMT'14 En-Fr: +", round(34.8 - 33.3, 1), "BLEU")
print("Transformer 41.8 BLEU on En-Fr, trained 3.5 days on 8 GPUs = ", 3.5 * 8 * 24, "GPU-hours")
print("BERT GLUE 80.5%, a 7.7 point gain, so the old best was", round(80.5 - 7.7, 1), "%")
print("GPT-3: 175B parameters. LLaMA-13B beats it on most benchmarks (abstract): size ratio", round(13 / 175, 3), "=", round(175 / 13, 1), "x smaller")
print("word2vec: 1.6 billion words in less than a day (abstract) -> at least", round(1.6e9 / 86400), "words per second")
```

## 🔴 4. Deep / formal
This note has no equation, so no worked formula. The patterns that repeat:
- **A limit becomes the next paper's first sentence.** Perceptron: one line only. Deeper network: hard to train. Deep plain stack: harder to train ([[ResNet 2015]]). Recurrence: slow and forgetful ([[Attention Is All You Need 2017]]). Dense Transformer: too costly ([[Mixture of Experts]]).
- **Three ingredients keep trading places:** better architectures, more data, more compute. The [[Scaling Laws]] work tries to say how they trade.
- **Every gain has a price,** often hidden in the abstract: attention made the cost grow with the square of length; diffusion needs 1000 steps; reasoning models write longer answers ([[Reasoning Model]]).
- **Caution on the story itself.** A tidy timeline hides parallel work, failed ideas and credit disputes (for example backpropagation was found several times independently, see [[Backpropagation]]). Treat years as pointers, and open the primary paper before quoting.
- **Not covered:** symbolic AI, expert systems, support vector machines and boosting, speech, robotics, game-playing, and the safety debate.

**What came before, and how this improved on it:**
- **Before:** hand-written rules and logic.
- **Its limit:** the world was too messy to write down, and rules could not learn.
- **What changed:** models learned from data, then learned their own features, then learned from huge unlabelled text.
- **How much better:** ImageNet top-5 error fell from 15.3% (2012) to 3.57% (2015), see the code and [[History of Image Classification]]; translation BLEU, GLUE and other numbers are in [[History of Text Analytics]].

## Related
- [[History of Image Classification]] · [[History of Text Analytics]] · [[From FNN to RNN]] · [[Neural Network]] · [[Machine Learning]] · [[Backpropagation]] · [[Transformer]] · [[LLM]] · [[Reinforcement Learning]] · [[Scaling Laws]] · [[Perceptron]]

## Sources
- Wikipedia, *Timeline of artificial intelligence* (read through a web fetch; the first 100,000 characters only, which ended partway through 2015)
- arXiv abstracts for 1406.2661, 1409.3215, 1409.0473, 1512.03385, 1706.03762, 1810.04805, 2005.14165, 2302.13971 and 2501.12948 (read)
- The vault's own checked notes: [[LeNet and AlexNet]], [[ResNet]]
- Dates marked "memory" above were not checked.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/history History of AI:::Three big stages, each fixing a limit of the one before: rules and logic, learning from data, then deep learning, followed by scale and language models.

#flashcards/ml/reverse #flashcards/topic/history Rules and logic (1950s to 1980s):::Humans write the knowledge down by hand. Good for narrow, clear problems. It fails when the world is messy, as in seeing or talking.

#flashcards/ml/reverse #flashcards/topic/history Learning from data (1980s to 2000s):::A machine learning model finds patterns in examples. Humans still design the features.

#flashcards/ml/reverse #flashcards/topic/history Deep learning (2010s on):::A many-layer neural network learns the features too, trained by backpropagation on GPUs. It beat hand-made features first in pictures, then in language.

#flashcards/ml/reverse #flashcards/topic/history Scale and language models (2017 on):::The Transformer made training parallel, and pre-training on huge text produced LLMs. RLHF and then reasoning models shaped what they do.

#flashcards/ml/concept #flashcards/topic/history What limit of rule-based AI did learning from examples fix?::Hand-written rules worked for narrow games and puzzles but not for seeing or talking. Letting the computer learn from examples fixed that.

#flashcards/ml/concept #flashcards/topic/history What limit of classic machine learning did deep learning fix?::Humans still had to design the features. A many-layer network learns the features itself.

#flashcards/ml/concept #flashcards/topic/history Name the two other threads that run through the history of AI.::The sequence thread (from plain networks to RNNs and attention) and the learning-by-trying thread (reinforcement learning).
