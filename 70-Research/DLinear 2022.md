---
tags: [paper, time-series, forecasting, transformer, linear, baseline]
link: https://arxiv.org/abs/2205.13504
authors: Zeng, Chen (equal first authors), Zhang, Xu (Chinese University of Hong Kong and IDEA; from the paper's first page)
year: 2022
status: body-read
field: time series forecasting
---

# DLinear 2022

> **In one line for a 12-year-old:** Many people built fancy, complicated "Transformer" programs to forecast long stretches of the future. These authors tried a very plain method that just takes a weighted sum of the past numbers, and it did better than the fancy ones on the usual test sets.

Paper: "Are Transformers Effective for Time Series Forecasting?", arXiv 2205.13504. First posted on 26 May 2022 (arXiv listing). I read version 3 of 17 August 2022 (paper's first page). From memory (not in the text I read): it appeared at AAAI 2023. "DLinear" is the best-known of its simple models, which the paper calls LTSF-Linear.

## The problem
A wave of [[Transformer]]-based models had appeared for **long-term time series forecasting** (LTSF: predicting many steps ahead): LogTrans, Informer, Autoformer, Pyraformer and FEDformer (paper). The authors question this trend. [[Self-Attention]] is permutation-invariant: it does not care about the order of its inputs unless positional information is added, and for a time series the order is the whole point. They also point out that the non-Transformer baselines in those papers forecast step by step, feeding each prediction back in (**iterated multi-step**, IMS), which is known to accumulate errors over long horizons (paper). See [[Time Series]] and the multi-step discussion in [[Forecasting Shoot-out]].

## What people did before
- **Statistical and classic ML:** [[ARIMA]] and gradient boosted trees, then deep learning: [[Recurrent Neural Network]] and temporal convolutional networks (paper).
- **Transformer variants for LTSF,** each adding time-series tricks to attention: sparse attention (LogTrans, Informer, Pyraformer), series decomposition inside the network (Autoformer, FEDformer), frequency-domain blocks (FEDformer), and special decoders that output the whole horizon at once ("direct multi-step", DMS) (paper).

## The idea
1. **Use direct multi-step forecasting for the baseline too.** Predict all $T$ future steps at once from the last $L$ values, so there is no error accumulation (paper, section 2).
2. **The simplest model:** one linear layer along the **time axis**. For each variable separately, the forecast is a weighted sum of the look-back window, $\hat X=WX$, where $W$ is a $T\times L$ matrix, with the weights shared across all variables and no model of the links between variables (paper, section 4). In the spirit of [[Linear Regression]] on lags.
3. **DLinear:** first split the input window into a **trend** (a moving average) and a **remainder** (the seasonal part), the same decomposition idea as in [[Trend and Seasonality]]. Then apply one linear layer to each part and add the two outputs (paper).
4. **NLinear:** subtract the last value of the window from the whole window, apply one linear layer, and add the last value back. This is a simple fix for a level that shifts between training and test (paper).
5. **A repeat baseline:** "Closest Repeat" just repeats the last value of the window, the naive forecast ([[Forecast Metrics]]) (paper).
6. **Compare against the five Transformers** on nine standard benchmarks: ETTh1, ETTh2, ETTm1, ETTm2, Traffic, Electricity, Exchange-Rate, Weather and ILI, measured with mean squared error (MSE) and mean absolute error (MAE) (paper).

## What they found
Each number is marked "abstract" or "paper" (read in the body text, as extracted from the PDF).
- **Headline:** LTSF-Linear outperforms the sophisticated Transformer models in all cases, often by a large margin (abstract). The paper gives 20% to 50% (introduction and section 5.2).
- **An example (paper, Table 2, Electricity, horizon 96):** DLinear MSE 0.140 and MAE 0.237, against FEDformer 0.193 and 0.308, Autoformer 0.201 and 0.317, Informer 0.274 and 0.368. The table's improvement figure for this row is 27.40%. For horizon 720 the improvement is 17.47% (paper).
- **The naive repeat can beat them:** on Exchange-Rate, Closest Repeat beat all the Transformer-based methods by around 45%. The authors say the Transformers predict the trend wrongly and overfit to sudden changes in the training data (paper).
- **Longer history does not help Transformers:** with look-back windows from 24 to 720 steps and a horizon of 720, the Transformers' errors stayed the same or got worse as the window grew, while all the LTSF-Linear models improved clearly. The authors conclude that the Transformers overfit temporal noise when given a longer sequence (paper, section 5.3).
- **Turning Informer into a linear model makes it better (paper, Table 4, MSE):** on Exchange-Rate at horizon 96, Informer 0.847, with attention replaced by a linear layer 1.003, embeddings plus a linear layer 0.173, and a single linear layer 0.084. On ETTh1 at horizon 96: 0.865, 0.613, 0.454 and 0.400. So the complicated parts did not help on these benchmarks.
- **They barely use the order (paper, Table 5):** when the input sequence is randomly shuffled, the average increase in error on ETTh1 is 81.06% for the linear model, 73.28% for FEDformer and 56.91% for Autoformer, but only 1.98% for Informer. On Exchange-Rate the Transformers' error did not change at all under shuffling. The authors read this as the Transformers preserving little temporal order, while a model that really uses order suffers when the order is destroyed.
- **The embeddings do carry some order:** removing Informer's positional embeddings makes its error rise sharply, and removing its timestamp embeddings damages it more as the horizon grows (paper, text on Table 6; I do not quote the table values).

## Why it mattered
It became a standard warning for forecasting research: a one-layer linear baseline must be included before claiming that a complex architecture helps, and long-horizon benchmark results from complex models should be checked against simple direct baselines. The authors themselves say their contribution is the question and the comparisons, not the linear model (paper, conclusion). From memory (not checked here): later Transformer variants (for example PatchTST) reported better results than DLinear by changing how the input is tokenised, so the story is not that attention can never work. See also [[N-BEATS 2019]] for a deep model that does well on a different kind of benchmark.

## Limits and open questions
- **Benchmarks only:** the conclusion is limited to "at least for the existing LTSF benchmarks" and nine datasets (paper). The authors say their results do not show that Transformers are useless for time series generally.
- **Limited capacity:** the linear model cannot capture changes caused by sudden shifts ("change points"), and the authors present it only as a simple, interpretable baseline (paper, conclusion).
- **Long horizons only:** the focus is on predicting many steps ahead (hundreds), not the short horizons of the airline-style examples in [[Forecasting Shoot-out]]. The authors hypothesise that long-term forecasting is only feasible for series with a fairly clear trend and periodicity (paper).
- **Not all time series are predictable:** chaotic series are an example the authors mention (paper).
- **Other tasks untested:** they advocate re-checking Transformers for other time series tasks such as anomaly detection, but do not do it (abstract).
- **My reading limits:** I did not verify the numbers by re-running the code, and I did not read the appendix, which contains the univariate results and extra experiments.

## Concepts used
- [[Time Series]] · [[Trend and Seasonality]] · [[Forecast Metrics]] · [[Linear Regression]] · [[Transformer]] · [[Self-Attention]] · [[Positional Encoding]] · [[Recurrent Neural Network]] · [[Overfitting]] · [[ARIMA]] · [[Gradient Boosting]] · [[Time Series Cross-Validation]]

## Reading trail
- **Came from:** [[Attention Is All You Need 2017]] (the architecture being questioned).
- **Led to:** no paper in this vault yet. [[N-BEATS 2019]] is an earlier, related deep forecasting model that is not part of its argument.

## How much I read
I downloaded the full paper (15 pages) and read the extracted text of: the abstract, the introduction, the problem formulation, the description of the Transformer-based solutions, the linear baselines, the experiments (Tables 1 to 5 and the analyses in section 5.3, with Table 6 and Table 8 only through their captions and discussion) and the conclusion.

**Not read:** the appendix (non-Transformer related work, detailed settings, more look-back window experiments, univariate results and visualisations). Figures were read only through their captions and the text around them. Table values come from text extracted from the PDF in column order; I matched each to its row, and I quote only the ones whose meaning is stated in the paper's text. I did not re-run any experiment, and the AAAI 2023 venue is from memory. The note was not opened in Obsidian.

## Sources
- Zeng, Chen, Zhang & Xu, "Are Transformers Effective for Time Series Forecasting?", arXiv 2205.13504 (v1 26 May 2022, v3 17 August 2022): https://arxiv.org/abs/2205.13504

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-time-series DLinear (Zeng, Chen, Zhang, Xu, 2022, arXiv 2205.13504): what question did it ask? (paper)::Are Transformers really effective for long-term time series forecasting? Self-attention is permutation-invariant, but in a time series the order is the whole point. The baselines in earlier Transformer papers also forecast step by step, which accumulates errors.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what is the simplest model, LTSF-Linear? (paper)::One linear layer along the time axis that directly predicts all T future steps from the last L values, X-hat = W X with W of size T by L, with weights shared across variables and no model of links between variables.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what is the difference between DLinear and NLinear? (paper)::DLinear splits the input into a trend (moving average) and a remainder, applies one linear layer to each and adds the outputs. NLinear subtracts the window's last value, applies one linear layer, and adds the last value back, which helps when the level shifts between training and test.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what was the headline result? (abstract, paper)::LTSF-Linear outperformed the sophisticated Transformer-based models in all cases on nine datasets, often by a large margin (20% to 50% in the paper). Example, Electricity at horizon 96: DLinear MSE 0.140 against FEDformer 0.193 (improvement 27.40% in the paper's table).

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what happened on Exchange-Rate with the naive repeat baseline? (paper)::Closest Repeat (repeat the last value) beat all Transformer-based methods by around 45%. The authors say the Transformers predict the trend wrongly and overfit to sudden changes in the training data.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what did the look-back window experiment show? (paper)::With windows from 24 to 720 steps and horizon 720, the Transformers' errors stayed the same or got worse as the window grew, while all LTSF-Linear models improved clearly. The authors conclude the Transformers overfit temporal noise when given longer sequences.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what did turning Informer into a linear model show? (paper, MSE)::On Exchange-Rate at horizon 96: Informer 0.847, attention replaced by linear 1.003, embeddings plus linear 0.173, one linear layer 0.084. On ETTh1 at horizon 96: 0.865, 0.613, 0.454 and 0.400. Simplifying made it better.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what did the shuffle test show? (paper)::Shuffling the input on ETTh1 raised the error by 81.06% for the linear model, 73.28% for FEDformer, 56.91% for Autoformer and only 1.98% for Informer. On Exchange-Rate the Transformers' error did not change. The authors read this as the Transformers preserving little temporal order.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what are its limits? (paper)::Conclusions apply to the existing benchmarks (nine datasets), not Transformers in general. The linear model has limited capacity and cannot capture change points. The focus is long horizons, long-term forecasting is only feasible with clear trend and periodicity, and other tasks such as anomaly detection were not tested.

#flashcards/research/concept #flashcards/topic/research-time-series DLinear: what is its place in the reading trail?::It came from Attention Is All You Need (the architecture it questions). No paper in the vault follows from it. N-BEATS 2019 is an earlier, related deep forecasting model that is not part of its argument.

#flashcards/research/concept #flashcards/topic/research-time-series How much of the DLinear paper did the note read, and what was not read?::The abstract, introduction, formulation, description of the Transformer solutions, the linear baselines, the experiments (Tables 1 to 5 and the analyses) and the conclusion. Not read: the appendix. Figures only through captions. No experiment re-run. The AAAI 2023 venue is from memory.
