---
tags: [BLEU, evaluation, machine-translation, NLP, metric, n-gram, ML]
status: growing
level: 4
tier: 2
---

# BLEU Score

**BLEU** (bilingual evaluation understudy) is a number that says how close a machine translation is to one or more human translations. It counts how many of the machine's word groups (**n-grams**) also appear in the human versions, then scales down answers that are too short. It runs in seconds and costs nothing, which is why papers such as [[Attention Is All You Need 2017]] report it. The original paper is Papineni, Roukos, Ward and Zhu, ACL 2002 (checked on the ACL Anthology page).

## 🟢 1. Simple (age 12)
A teacher marks your translation by checking how many of your words and little phrases also appear in the answer sheets written by good translators. The more of your phrases match, the higher your mark. If you write only two words, the teacher takes marks off, because it is too easy to match with so few words.

**Example:** The answer sheet says "the cat is on the mat". If you write "the the the the the the the", every word is on the sheet, but you only get credit for the number of times "the" really appears there (twice), so your mark is low.

## 🔵 2. How it works
1. **Get references.** Collect one or more good human translations of each sentence.
2. **Cut into n-grams.** An n-gram is a run of $n$ words in a row: single words (1-grams), pairs (2-grams), triples and fours.
3. **Count matches, with a limit ("clipping").** For each n-gram of the machine's sentence, count it as a match at most as many times as it appears in *one* reference. This stops "the the the the" from cheating.
4. **Turn counts into precision.** For each size $n$ divide the clipped matches by the number of n-grams the machine wrote. Call the result $p_n$.
5. **Combine with a geometric average.** Multiply the four precisions and take the fourth root. One zero makes the whole score zero.
6. **Punish short answers.** Multiply by a brevity penalty that is 1 for a long enough answer and drops fast for a short one. (Long answers are already hurt by the precision step.)
7. **Read the score.** It is between 0 and 1 (often shown times 100). 1 would mean identical to a reference.

I found no free image of how BLEU works on Wikimedia Commons, so there is no figure here; the worked numbers below do that job.

## 🟠 3. Technical detail
All numbers come from the code at the end of this section (Python 3.9.6, SacreBLEU 2.4.0). I wrote BLEU from the paper's formulas and checked it against SacreBLEU on the paper's own Example 1 (one machine sentence, three human references, no tokenisation, no smoothing, lower-case).

- **The paper's numbers reproduced.** Candidate 1 ("It is a guide to action which ensures that the military always obeys the commands of the party") has a clipped unigram precision 17/18 and bigram precision 10/17, exactly the paper's. Candidate 2 ("It is to insure the troops forever hearing the activity guidebook that party direct") has 8/14 and 1/13, also the paper's. The paper's Example 2 ("the the the the the the the") gives 2/7 against a plain precision of 7/7.
- **Scores.** Candidate 1: 3-gram precision 7/16, 4-gram precision 4/15, brevity penalty 1, BLEU 0.5046, and SacreBLEU printed the same. Candidate 2 has no matching 3-gram or 4-gram, so its BLEU is 0.
- **Same words, wrong order.** Candidate 1 written backwards keeps a unigram precision of 17/18 (0.944) but has a bigram precision of 0.059 and no matching 3-grams, so BLEU is 0.
- **Another way of saying it.** "It is a guide for action which ensures that the army always follows the orders of the party" (my own rewording with synonyms) scored 0.3231. It is a good translation that BLEU punishes because the words differ from the references.
- **More references, higher score.** The same candidate scored 0.4209 against reference 1 alone and 0.5046 against all three. The paper makes the same point with human translators: 0.3468 against four references and 0.2571 against two.
- **Corpus, not sentence.** Two sentences scored 1.0 and 0.0 alone (the second is too short to have any 4-gram), but together 0.6065, because the brevity penalty is computed over the whole corpus (the paper's section 2.2.2). Averaging sentence scores is not BLEU.
- **Limits of my run:** one sentence set from the paper, no real system output, no other tokenisers, no smoothing. SacreBLEU's smoothing and 13a tokeniser were switched off, so my numbers do not match what you would get on real data with default settings.

```python
import math
from collections import Counter
import sacrebleu

def ngrams(t, n): return Counter(tuple(t[i:i+n]) for i in range(len(t) - n + 1))

def clipped(cand, refs, n):
    """clipped matches, total candidate n-grams (paper, 2.1)"""
    c, mx = ngrams(cand, n), Counter()
    for r in refs:
        for g, k in ngrams(r, n).items(): mx[g] = max(mx[g], k)
    return sum(min(k, mx[g]) for g, k in c.items()), sum(c.values())

def bleu(cands, refs_list, N=4):
    m, t, c, r = [0] * N, [0] * N, 0, 0
    for cand, refs in zip(cands, refs_list):
        cand = cand.lower().split(); refs = [x.lower().split() for x in refs]
        for n in range(1, N + 1):
            a, b = clipped(cand, refs, n); m[n-1] += a; t[n-1] += b
        c += len(cand); r += min((abs(len(x) - len(cand)), len(x)) for x in refs)[1]   # closest reference length
    p = [a / b if b else 0 for a, b in zip(m, t)]
    bp = 1 if c > r else math.exp(1 - r / c)
    return (0.0 if min(p) == 0 else bp * math.exp(sum(math.log(x) / N for x in p))), p, bp

c1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
c2 = "It is to insure the troops forever hearing the activity guidebook that party direct"
refs = ["It is a guide to action that ensures that the military will forever heed Party commands",
        "It is the guiding principle which guarantees the military forces always being under the command of the Party",
        "It is the practical guide for the army always to heed the directions of the party"]
for name, c in (("cand1", c1), ("cand2", c2)):
    s, p, bp = bleu([c], [refs])
    print(name, 'BLEU', round(s, 4), 'p', [round(x, 4) for x in p], 'BP', round(bp, 4))
print('sacrebleu cand1', round(sacrebleu.corpus_bleu([c1], [[r] for r in refs], tokenize='none', smooth_method='none', lowercase=True).score / 100, 4))
w = ["the"] * 7; rr = ["the cat is on the mat".split(), "there is a cat on the mat".split()]
print('example 2 clipped unigrams:', clipped(w, rr, 1))
print('reversed:', [round(x, 3) for x in bleu([" ".join(reversed(c1.split()))], [refs])[1]])
syn = "it is a guide for action which ensures that the army always follows the orders of the party"
print('synonyms:', round(bleu([syn], [refs])[0], 4))
print('refs 1 vs 3:', round(bleu([c1], [refs[:1]])[0], 4), round(bleu([c1], [refs])[0], 4))
A = ["the cat sat on the mat today", "a dog barks"]; R = [["the cat sat on the mat today"], ["a dog barks loudly at night every day"]]
print('corpus', round(bleu(A, R)[0], 4), '| each alone', [round(bleu([a], [r])[0], 4) for a, r in zip(A, R)])
```

## 🔴 4. Deep / formal
**Modified (clipped) n-gram precision** (paper, section 2.1), over a whole test corpus:

$$
p_n = \frac{\sum_{C \in \text{Candidates}} \sum_{g \in C} \text{Count}_{\text{clip}}(g)}{\sum_{C \in \text{Candidates}} \sum_{g \in C} \text{Count}(g)}, \qquad \text{Count}_{\text{clip}}(g) = \min\big(\text{Count}(g),\ \text{MaxRefCount}(g)\big)
$$

**Where:**
- $C$ is one machine-translated sentence from the test set. **Matters:** matches are counted sentence by sentence and then summed, so one long sentence cannot hide many short bad ones.
- $g$ is one n-gram (a run of $n$ words) of that sentence.
- $\text{Count}(g)$ is how many times $g$ appears in $C$.
- $\text{MaxRefCount}(g)$ is the largest number of times $g$ appears in any *single* reference. **Matters:** this is the clip. Without it, "the the the the the the the" would score 7/7 instead of 2/7.
- $p_n$ is the share of the machine's n-grams that are credited. **Matters:** it is the main quality signal.

**Example (tiny numbers):** the paper's Example 2: candidate "the the the the the the the" has $\text{Count}(\text{the})=7$. In the references "the" appears twice in one ("The cat is on the mat") and once in the other, so $\text{MaxRefCount}=2$. Then $\text{Count}_{\text{clip}}=\min(7,2)=2$ and $p_1=2/7=0.286$. The answer means: only two of the seven words are credited; plain precision would have said a perfect 7/7.

**Brevity penalty** (section 2.3), with $c$ the total length of the machine's corpus and $r$ the effective reference length (for each sentence, the length of the closest reference, summed):

$$
\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{\,1 - r/c} & \text{if } c \le r \end{cases}
$$

**Where:**
- $c$ is the number of words the system wrote in total.
- $r$ is the number of words the closest references have in total. **Matters:** it compares with the references, not with the source sentence.
- $\text{BP}$ is the multiplier. **Matters:** precision alone loves very short outputs (a single correct word gives $p_1=1$); the penalty removes that cheat. Long outputs are already hurt by precision, so they get no extra penalty.

**Example (tiny numbers):** the system wrote $c=8$ words and the closest references have $r=12$. Then $r/c=1.5$ and $\text{BP}=e^{1-1.5}=e^{-0.5}=0.6065$. With $c=12$, $r=12$ the penalty is 1. With $c=1$, $r=12$ it would be $e^{-11}=0.0000167$. The answer means: writing two-thirds of the length cuts the score by about 40%.

**BLEU** (section 2.3), with $N=4$ and weights $w_n=1/4$ in the paper's baseline:

$$
\text{BLEU} = \text{BP}\cdot \exp\!\Big(\sum_{n=1}^{N} w_n \log p_n\Big)
$$

**Where:**
- $N$ is the longest n-gram size used (4 in the baseline).
- $w_n$ is the weight of size $n$; they add up to 1 (uniform: 1/4 each). **Matters:** it decides how much one-word matches count against four-word matches.
- $\log p_n$ is the log of each precision. **Matters:** the weighted sum of logs, then $\exp$, is the geometric mean. It is harsh by design: one $p_n=0$ makes BLEU zero (the paper's footnote 5 says this).
- $\text{BP}$ is the brevity penalty above.

**Example (tiny numbers):** let $p=(0.8,\,0.4,\,0.2,\,0.1)$ and $c=9$, $r=10$. The geometric mean is $(0.8\cdot0.4\cdot0.2\cdot0.1)^{1/4}=0.0064^{1/4}=0.2828$. The penalty is $e^{1-10/9}=0.8948$. So BLEU $=0.8948\times0.2828=0.2531$ (my arithmetic). The answer means: 25 BLEU points, about 3 of 100 lost to being slightly short.

**The paper's own evidence (its Table 1 and Table 2).** The paper scored five systems on 500 Chinese-to-English sentences with two references:

| System | BLEU (paper's Table 1) | Mean over 20 blocks (Table 2) | Standard deviation | Paired t against the left neighbour |
|---|---|---|---|---|
| S1 | 0.0527 | 0.051 | 0.017 | none |
| S2 | 0.0829 | 0.081 | 0.025 | 6 |
| S3 | 0.0930 | 0.090 | 0.020 | 3.4 |
| H1 | 0.1934 | 0.192 | 0.030 | 24 |
| H2 | 0.2571 | 0.256 | 0.039 | 11 |

Table written by me, an AI, from the paper's two tables (text extraction; I read the numbers from the extracted text and did not look at the page picture). I merged the two paper tables into one.
- **System:** S1 to S3 are three machine translation systems; H1 and H2 are two human translators (the paper does not name the systems). **Matters:** it shows that BLEU puts humans far above machines.
- **BLEU (Table 1):** BLEU over all 500 sentences.
- **Mean and standard deviation (Table 2):** the 500 sentences were split into 20 blocks of 25 sentences, BLEU was computed per block, and these are the mean and spread over the 20. **Matters:** they show how much BLEU moves between small test sets.
- **t:** the paired t-statistic comparing each system with its left neighbour; the paper says a value of 1.7 or more is significant at 95%.

**Summary: what the table shows.** BLEU orders the five systems S1 < S2 < S3 < H1 < H2, and every difference is significant, including the small one between S2 and S3 (t = 3.4). **Caution:** the paper compared BLEU with human judgments on only these five systems, one language pair and 250 sentence pairs (paper, section 4), and I did not read its human-judgment figures as numbers.

**What is wrong with BLEU (limits).**
- It does not see meaning: my rewording with synonyms lost about 18 points (0.5046 to 0.3231) though it is a good translation.
- It depends on the number of references and on tokenisation, so scores from different set-ups are not comparable (paper: 0.3468 against four references and 0.2571 against two for the same human). SacreBLEU exists to fix the tokenisation part (general knowledge).
- Scores mean different things on different language pairs; do not compare English-German with English-French as "quality" (general knowledge).
- Modern text generation (chat, summaries) has no single correct answer; people use [[Perplexity]], [[Benchmarks]], human preference ([[Chatbot Arena]]) or [[LLM as a Judge]] instead (general knowledge).

**What came before, and how this improved on it:**
- **Before:** human evaluation, which the paper says is "quite expensive" and can "take weeks or months to finish" (paper, 1.1). The paper models its metric on the word error rate used in speech recognition.
- **Its limit:** developers could not test daily changes to their systems.
- **What changed:** a cheap, automatic, language-independent closeness score against reference translations.
- **How much better:** the paper's tests show BLEU's differences between its five systems are statistically significant (its Table 2, above), and its text says BLEU correlates highly with the monolingual judges and estimates even the small S2 against S3 difference well (paper, section 5, Figure 7; I did not read the figure as numbers). Its conclusion also cites a separate analysis in four languages (Arabic, Chinese, French, Spanish) that I did not read. I did not compute any correlation myself. Today it appears in results such as the Transformer's 28.4 BLEU English-German and 41.8 English-French (from my [[Attention Is All You Need 2017]] note).

## Related
[[Evaluation Metrics]] · [[Precision]] · [[Perplexity]] · [[Benchmarks]] · [[Chatbot Arena]] · [[LLM as a Judge]] · [[Beam Search]] · [[Tokenization]] · [[Attention Is All You Need 2017]] · [[Seq2Seq 2014]]

## Sources
- Papineni, K., Roukos, S., Ward, T., Zhu, W.-J. (2002). *Bleu: a Method for Automatic Evaluation of Machine Translation.* Proceedings of the 40th Annual Meeting of the ACL, pp. 311 to 318. Page checked: <https://aclanthology.org/P02-1040/> (title, authors, date, pages, DOI 10.3115/1073083.1073135). I downloaded the PDF (8 pages) and read it as extracted text, in full except the figures and the reference list. Figures 3 and 7 were not read as numbers.
- My own run (Python 3.9.6, SacreBLEU 2.4.0). Not checked: the note was not opened in Obsidian; no real translation system was scored.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation BLEU:::A score from 0 to 1 for how closely a machine translation matches one or more human translations, by counting shared word groups (n-grams) and punishing too-short outputs.

#flashcards/ml/concept #flashcards/topic/evaluation Why is BLEU's precision "modified" or "clipped"?::Each candidate n-gram counts at most as many times as it appears in a single reference. Without it, "the the the the the the the" would score 7/7; with clipping it scores 2/7 (the paper's Example 2).

#flashcards/ml/concept #flashcards/topic/evaluation Why does BLEU have a brevity penalty?::Precision alone favours very short outputs (one correct word gives precision 1). The penalty $e^{1-r/c}$ cuts the score when the output is shorter than the references. Long outputs need no extra penalty because precision already hurts them.

#flashcards/ml/concept #flashcards/topic/evaluation Why does BLEU use a geometric mean of the n-gram precisions, and what is the consequence?::The paper takes the geometric mean; the consequence is that a precision of zero for any $n$ makes BLEU zero. In my run, candidate 2 of the paper's Example 1 had no matching 3-gram, so its BLEU was 0.

#flashcards/ml/formula #flashcards/topic/evaluation Write BLEU's modified n-gram precision and explain each symbol.
?
$$p_n = \frac{\sum_{C} \sum_{g \in C} \text{Count}_{\text{clip}}(g)}{\sum_{C} \sum_{g \in C} \text{Count}(g)}, \qquad \text{Count}_{\text{clip}}(g) = \min\big(\text{Count}(g),\ \text{MaxRefCount}(g)\big)$$
- $C$: one machine-translated sentence; matches are counted per sentence and then summed over the corpus.
- $g$: one n-gram (a run of $n$ words).
- $\text{Count}(g)$: how often $g$ appears in $C$.
- $\text{MaxRefCount}(g)$: the largest number of times $g$ appears in any single reference. This is the clip; without it repeated words cheat.
- $p_n$: the share of the machine's n-grams that are credited.

#flashcards/ml/example #flashcards/topic/evaluation Candidate "the the the the the the the" against references "The cat is on the mat" and "There is a cat on the mat": what is $p_1$?::"the" appears 7 times in the candidate and at most 2 times in one reference, so the clipped count is $\min(7,2)=2$ and $p_1=2/7=0.286$ (the paper's Example 2).

#flashcards/ml/formula #flashcards/topic/evaluation Write BLEU's brevity penalty and explain each symbol.
?
$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{\,1 - r/c} & \text{if } c \le r \end{cases}$$
- $c$: total number of words the system wrote.
- $r$: total length of the closest reference for each sentence, summed.
- $\text{BP}$: the multiplier. It removes the cheat of writing very little; long outputs are not punished again.

#flashcards/ml/example #flashcards/topic/evaluation The system writes $c=8$ words and the closest references have $r=12$. What is the brevity penalty?::$r/c=1.5$, so $\text{BP}=e^{1-1.5}=e^{-0.5}=0.6065$ (my arithmetic). For $c=12$ it is 1.

#flashcards/ml/formula #flashcards/topic/evaluation Write the BLEU formula and explain each symbol.
?
$$\text{BLEU} = \text{BP}\cdot \exp\!\Big(\sum_{n=1}^{N} w_n \log p_n\Big)$$
- $N$: the longest n-gram size (4 in the paper's baseline).
- $w_n$: the weight of size $n$; they sum to 1 (1/4 each in the baseline).
- $\log p_n$: the log precision; the weighted sum then $\exp$ is the geometric mean, so one zero precision makes BLEU zero.
- $\text{BP}$: the brevity penalty above.

#flashcards/ml/example #flashcards/topic/evaluation With $p=(0.8,0.4,0.2,0.1)$, equal weights, $c=9$ and $r=10$, what is BLEU?::Geometric mean $=(0.0064)^{1/4}=0.2828$; $\text{BP}=e^{1-10/9}=0.8948$; BLEU $=0.8948\times0.2828=0.2531$ (my arithmetic).

#flashcards/ml/concept #flashcards/topic/evaluation Why does the weight of a $\log p_n$ term matter, and what does equal weighting say?::The weights decide how much short matches count against long matches. The paper's baseline uses $N=4$ and $w_n=1/4$, so a one-word match counts as much as a four-word match in the log.

#flashcards/ml/concept #flashcards/topic/evaluation Why is BLEU computed over a whole corpus rather than averaged per sentence?::The brevity penalty is taken over the corpus (paper, 2.2.2) so that length differences on short sentences are not punished harshly. In my run two sentences scored 1.0 and 0.0 alone but 0.6065 together.

#flashcards/ml/concept #flashcards/topic/evaluation What does the paper's Example 1 show about the quality of the two candidates in numbers?::Candidate 1 has unigram precision 17/18 and bigram 10/17; candidate 2 has 8/14 and 1/13 (all the paper's numbers, reproduced in my run). My BLEU: 0.5046 for candidate 1 and 0 for candidate 2.

#flashcards/ml/concept #flashcards/topic/evaluation What is a weakness of BLEU shown by rewording a good translation with synonyms?::BLEU only sees matching words, not meaning. My synonym rewording of candidate 1 dropped from 0.5046 to 0.3231 although it is still a good translation.

#flashcards/ml/concept #flashcards/topic/evaluation Why can BLEU scores from two papers not always be compared?::BLEU depends on the number of references (the paper: 0.3468 with four references against 0.2571 with two for one human translator) and on tokenisation, so different set-ups give different numbers.

#flashcards/ml/cloze #flashcards/topic/evaluation BLEU multiplies the geometric mean of the n-gram precisions by a ==brevity penalty==, and the paper's baseline uses n-grams up to length ==4==.

#flashcards/ml/concept #flashcards/topic/evaluation What problem with earlier evaluation did the BLEU paper try to fix?::Human evaluation was "quite expensive" and could "take weeks or months" (paper, 1.1), so developers could not check daily changes. BLEU is cheap, fast and language-independent.
