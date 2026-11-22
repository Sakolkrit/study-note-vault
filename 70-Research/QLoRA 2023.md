---
tags: [paper, llm, fine-tuning, quantization, lora, parameter-efficient]
link: https://arxiv.org/abs/2305.14314
authors: Dettmers, Pagnoni (equal first authors), Holtzman, Zettlemoyer (University of Washington; from the paper's first page)
year: 2023
status: body-read
field: large language models, fine-tuning
---

# QLoRA 2023

> **In one line for a 12-year-old:** Squeeze a huge language model down to 4-bit numbers so it fits on one computer, freeze it, and teach it new things by training only tiny add-on pieces. The squeezed model learns just as well, and this made it possible to customise giant models on a single graphics card.

Paper: "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv 2305.14314, first posted on 23 May 2023 (arXiv listing; the first page says "Preprint. Under review"). Concept note: [[QLoRA]]. It builds directly on [[LoRA 2021]].

## The problem
Fine-tuning a very large model is prohibitively expensive: regular 16-bit fine-tuning of a 65-billion-parameter LLaMA model needs more than 780 GB of GPU memory (paper). Quantization (storing the weights with fewer bits, see [[Quantization]]) had been used to shrink models, but the paper says such techniques "only work for inference and break down during training". Could you fine-tune a model that is stored in 4 bits without losing quality? (paper)

## What people did before
- **Full 16-bit fine-tuning:** the memory cost above (paper).
- **LoRA ([[LoRA 2021]]):** train small adapters on a frozen model. This removes most of the trainable-parameter and optimizer memory, but the frozen 16-bit model itself is still large (paper, background).
- **Quantization for inference:** methods such as LLM.int8() and SmoothQuant manage outlier values to keep 16-bit quality, and rounding-based methods study the trade-off. The authors note that 4-bit inference quantization degrades quality relative to 16-bit (paper, section 4 and related work).

## The idea
1. **Store the frozen base model in 4 bits and compute in 16 bits.** Whenever a weight block is needed, it is converted back ("dequantized") to BFloat16 and the matrix multiplication runs in 16 bits. Gradients flow through the frozen 4-bit model into the [[LoRA]] adapters, and only the adapters get weight updates (paper).
2. **4-bit NormalFloat (NF4).** Pre-trained weights are roughly bell-shaped ([[Normal Distribution]]). NF4 places its 16 levels at the quantiles of a normal curve, so each level is used by about the same number of weights, and it includes an exact zero. The paper calls this information-theoretically optimal for normally distributed data (abstract and paper).
3. **Double quantization.** Weights are quantized in blocks of 64, each with its own scaling constant. Storing those constants in 32 bits costs 32/64 = 0.5 bits per weight. Double quantization quantizes the constants too (8-bit floats, blocks of 256), bringing the cost to 8/64 + 32/(64 x 256) = 0.127 bits per weight, a saving of 0.373 bits per weight, about 3 GB for a 65B model (paper).
4. **Paged optimizers.** Use NVIDIA unified memory to move optimizer state to CPU memory automatically when the GPU runs out of memory during a long-sequence memory spike, and bring it back when needed (paper).
5. **Adapters on every linear layer.** The authors found that LoRA on only the query and value matrices (the usual choice, as in the LoRA paper) could not match full fine-tuning for large models, and that adapters on **all** linear layers of the transformer blocks are needed. The adapter rank $r$ did not matter in their tests (paper, Figure 2 and text).
6. **Use it at scale.** They fine-tuned more than 1,000 models across eight instruction datasets, LLaMA and T5, and sizes from 80 million to 65 billion parameters, and released the Guanaco chatbot family (paper).

## What they found
Each number is marked "abstract" or "paper" (read in the body text, as extracted from the PDF).

**Memory (abstract and paper)**
- QLoRA reduces the average memory needed to fine-tune a 65B model from more than 780 GB to under 48 GB of GPU memory, without degrading runtime or predictive performance compared with a 16-bit fully fine-tuned baseline (paper, introduction).
- In a breakdown for a 7B LLaMA at batch size 1 with LoRA weights of about 0.2% of the model: the LoRA parameters take 26 MB, the input gradients 567 MB (18 MB per sequence with gradient checkpointing), and the 4-bit base model 5,048 MB. The authors conclude that gradient checkpointing matters and that shrinking the number of LoRA parameters saves little memory, so more adapters can be used (paper, section 2).

**4-bit data types (paper)**
- Mean perplexity on Pile Common Crawl over OPT, BLOOM, LLaMA and Pythia models from 125M to 13B (Table 2): Int4 34.34, Float4 (E2M1) 31.07, Float4 (E3M0) 29.48, NormalFloat4 with double quantization **27.41**. Lower is better.
- Double quantization gave only minor accuracy changes. Its value is finer control of the memory footprint, so that 33B and 65B models fit 24 GB and 48 GB GPUs (paper, Figure 3 text).

**Does 4-bit match 16-bit? (paper, Tables 3 and 4)**
- GLUE with RoBERTa-large: BF16 88.6, LoRA BF16 88.8, QLoRA Int8 88.8, QLoRA FP4 88.6.
- Super-NaturalInstructions (RougeL) with T5: for 3B, BF16 54.3, LoRA BF16 55.4, QLoRA NF4 with double quantization 55.3. For 11B, BF16 62.0, LoRA BF16 60.7, QLoRA NF4 with double quantization 60.9.
- MMLU, LLaMA 7B to 65B on Alpaca and FLAN v2, mean 5-shot accuracy (Table 4): BFloat16 53.0, Float4 52.2, NormalFloat4 with double quantization **53.1**. The authors say NF4 with double quantization fully recovers the 16-bit LoRA score, while FP4 lags by about one percentage point.
- The authors' summary: 4-bit QLoRA with NF4 matches 16-bit full fine-tuning and 16-bit LoRA on the academic benchmarks they used.

**Guanaco chatbots (abstract and paper)**
- Guanaco is QLoRA fine-tuning on a variant of the OASST1 dataset. On the Vicuna benchmark, as a percentage of ChatGPT's score judged by GPT-4 (Table 6): Guanaco 65B **99.3%** (41 GB), Guanaco 33B 97.8% (21 GB), Guanaco 13B 90.4% (10 GB), Guanaco 7B 87.0% (5 GB); Vicuna 13B (16-bit) 94.9% (26 GB), Bard 94.8%, GPT-4 114.5%. The 95% confidence intervals are about 4 to 5 percentage points, so many of these overlap.
- In a tournament judged by GPT-4 (Table 1, Elo ratings): GPT-4 1348, Guanaco 65B 1022, Guanaco 33B 992, Vicuna 13B 974, ChatGPT 966, Guanaco 13B 916, Bard 902, Guanaco 7B 879. By human ratings, Guanaco 65B and 33B have an expected win probability of about 30% against GPT-4.
- Training time: Guanaco 33B in under 12 hours on a 24 GB consumer GPU, and the 65B model in 24 hours on one professional GPU (abstract and paper).
- **Data quality over quantity (paper):** a 9,000-example dataset (OASST1) beat a 450,000-example subsample of FLAN v2 on chatbot performance.
- **MMLU and chatbot quality are not the same skill (paper, Tables 5 and 6):** fine-tuning on FLAN v2 gave the best MMLU score but the worst Vicuna score. For LLaMA 7B, 13B, 33B and 65B the untuned MMLU scores are 35.1, 46.9, 57.8 and 63.4 and the FLAN v2 scores 44.5, 51.4, 59.2 and 63.9, while some datasets lowered MMLU: for example 13B on Self-Instruct got 33.3.
- **Judging chatbots (paper):** GPT-4 and human rankings agree only moderately at the system level (Kendall tau 0.43, Spearman 0.55) and weakly per example (Fleiss kappa 0.25). Human annotators agreed with each other moderately (kappa 0.42). GPT-4 prefers whichever answer comes first in its prompt, and rates its own outputs higher than humans do (Elo 1348 against 1176). The authors conclude that GPT-4 is a cheap and reasonable alternative to human evaluation, but that current chatbot benchmarks are not trustworthy enough to measure chatbot performance accurately (abstract and paper).

## Why it mattered
It made fine-tuning of 33B to 65B models possible on a single GPU, which the authors call an equaliser for small teams, and showed that quality was not lost by storing the base model in 4 bits (paper, broader impacts). It also gave a widely used recipe (NF4, double quantization, paged optimizers, adapters on all layers). From memory (not checked in this paper): it was integrated into the Hugging Face and bitsandbytes tooling and became a standard way to fine-tune open models on limited hardware; the paper itself says the code and CUDA kernels were released and integrated into Hugging Face `transformers`. The paper's warnings about benchmarks and about using a model as judge (see [[LLM as a Judge]] and [[Benchmarks]]) are as influential as the method.

## Limits and open questions
- **Not shown at the largest sizes:** the authors did not establish that QLoRA matches full 16-bit fine-tuning at 33B and 65B, because of the cost (paper, limitations).
- **Evaluation limits:** only MMLU, the Vicuna benchmark and the OA benchmark. Others (BigBench, RAFT, HELM) were not tried. The authors say performance likely depends on how similar the fine-tuning data is to the benchmark, so one must be careful what is being evaluated (paper).
- **Chatbot evaluation is shaky:** wide confidence intervals, moderate agreement between raters, and order and self-preference biases in GPT-4 (paper). The qualitative study shows Guanaco giving confident wrong answers on obscure facts, for example a wrong singer and birth year (paper, section 6.1).
- **Limited safety evaluation:** a limited responsible-AI check, with a bias test on which the fine-tuned model scored lower than raw pretrained models, but no wide study (paper).
- **Choices not explored:** bit widths below 4 (for example 3-bit) and adapter methods other than LoRA were not tested (paper).
- **Speed:** paged optimizers kept the same training speed as regular ones at batch size 16 on 65B, but no hard measurements for other cases (paper). The paper does not claim QLoRA is faster; it claims lower memory.
- **Dual-use:** the authors note that fine-tuning can be abused for harm, and argue that broader access still has a net positive effect (paper).

## Concepts used
- [[QLoRA]] · [[LoRA]] · [[Quantization]] · [[Fine-tuning]] · [[Normal Distribution]] · [[Instruction Tuning]] · [[Gradient Checkpointing]] · [[Mixed Precision Training]] · [[LLM as a Judge]] · [[Benchmarks]] · [[Chatbot Arena]] · [[GPU]] · [[Cross-Entropy Loss]] · [[Perplexity]] · [[Adam Optimizer]] · [[RLHF]]

## Reading trail
- **Came from:** [[LoRA 2021]] (the adapters it trains), [[LLaMA 2023]] (the model family it fine-tunes) and [[GPT-3 2020]] (the scale of model that motivates it).
- **Led to:** no paper in this vault yet. Its Guanaco models and the evaluation methods are used in later work (from memory, not checked).

## How much I read
I downloaded the full paper (26 pages) and read the extracted text of: the abstract, the introduction, the background on quantization and LoRA, the QLoRA method, the comparison with standard fine-tuning (Tables 2 to 4), the chatbot experiments (setup, evaluation, Tables 5 to 7), the qualitative analysis (first part of section 6.1 and section 6.2), related work, the limitations and the broader impacts.

**Not read:** the appendices (experimental details, the exact NF4 values, the qualitative lemon-and-cherry examples after the first few, memory breakdown tables and extra results), and the rest of section 6.1 and the later tables. Figures were read only through their captions and the text around them. Table values come from text extracted from the PDF in column order, and I matched each to its row. I did not re-run any experiment. The note was not opened in Obsidian.

## Sources
- Dettmers, Pagnoni, Holtzman & Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv 2305.14314 (23 May 2023): https://arxiv.org/abs/2305.14314

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm QLoRA (Dettmers, Pagnoni, Holtzman, Zettlemoyer, 2023, arXiv 2305.14314): what problem did it tackle? (paper)::Regular 16-bit fine-tuning of a 65B LLaMA needs more than 780 GB of GPU memory, and quantization had worked only for inference. QLoRA asks whether a model stored in 4 bits can be fine-tuned without losing quality.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: how does the training work? (paper)::The frozen base model is stored in 4-bit NormalFloat and converted to BFloat16 when a weight block is used. Gradients flow through the frozen model into LoRA adapters, and only the adapters (in 16-bit) are updated.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what is 4-bit NormalFloat (NF4)? (paper)::A 4-bit data type whose 16 levels sit at the quantiles of a normal curve, so each level is used by about the same number of weights, with an exact zero. The paper calls it information-theoretically optimal for normally distributed weights.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what does double quantization save? (paper)::Quantizing the per-block scaling constants too. With blocks of 64 and 32-bit constants the cost is 32/64 = 0.5 bits per weight. With 8-bit constants in blocks of 256 it is 8/64 + 32/(64 x 256) = 0.127 bits, a saving of 0.373 bits per weight, about 3 GB for a 65B model.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what are paged optimizers? (paper)::They use NVIDIA unified memory to move optimizer state to CPU memory automatically when the GPU runs out of memory during memory spikes, and back when needed. At batch size 16 on 65B they gave the same speed as regular optimizers (no other hard measurements).

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: which layers need adapters? (paper)::All linear layers of the transformer blocks. LoRA on only the query and value matrices could not match full fine-tuning for large models. The adapter rank did not matter in their tests.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: how did the 4-bit data types compare? (paper)::Mean Pile Common Crawl perplexity: Int4 34.34, Float4 (E2M1) 31.07, Float4 (E3M0) 29.48, NormalFloat4 with double quantization 27.41 (lower is better). On MMLU, NF4 with double quantization (53.1) matched BFloat16 (53.0), while Float4 was about one point behind (52.2).

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: did 4-bit match 16-bit? (paper)::Yes on the academic benchmarks they used. For example GLUE (RoBERTa-large): BF16 88.6, QLoRA FP4 88.6. T5-3B RougeL: BF16 54.3, LoRA 55.4, QLoRA NF4 55.3. They did not show it at 33B and 65B for full 16-bit fine-tuning because of the cost.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what did Guanaco achieve? (abstract, paper)::On the Vicuna benchmark, as a share of ChatGPT's score judged by GPT-4: Guanaco 65B 99.3% (41 GB), 33B 97.8% (21 GB), 13B 90.4% (10 GB), 7B 87.0% (5 GB). The 33B model trains in under 12 hours on a 24 GB GPU. The 95% intervals are about 4 to 5 points, so many results overlap.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what did the paper find about data and benchmarks? (paper)::Data quality beat quantity: 9,000 OASST1 examples beat a 450,000-example FLAN v2 subsample on chatbot quality. FLAN v2 gave the best MMLU but the worst Vicuna score, so strong MMLU does not mean a strong chatbot.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: how reliable were the chatbot judges? (paper)::GPT-4 and humans agree only moderately at the system level (Kendall tau 0.43, Spearman 0.55) and weakly per example (Fleiss kappa 0.25). GPT-4 prefers the first answer in its prompt and rates its own outputs higher than humans (Elo 1348 against 1176). The authors call it a cheap, reasonable alternative to human evaluation, but say current chatbot benchmarks are not trustworthy.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what are its limits? (paper)::Not shown to match full 16-bit fine-tuning at 33B and 65B. Only three benchmarks. Shaky chatbot evaluation. Confident wrong answers on obscure facts. Limited safety evaluation. Bit widths below 4 and other adapter methods untested. It saves memory, not time.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what is its place in the reading trail?::It came from LoRA 2021 (the adapters), LLaMA 2023 (the models fine-tuned) and GPT-3 (the scale that motivates it). No paper note in this vault follows from it yet.

#flashcards/research/concept #flashcards/topic/research-llm How much of the QLoRA paper did the note read, and what was not read?::The abstract, introduction, background, method, Tables 2 to 7, the first qualitative examples, section 6.2, related work, limitations and broader impacts. Not read: the appendices (exact NF4 values, memory breakdowns, extra results) and the rest of section 6.1. Figures only through captions. No experiment re-run.
