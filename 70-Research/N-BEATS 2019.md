---
tags: [paper, time-series, forecasting, deep-learning, residual, interpretable]
link: https://arxiv.org/abs/1905.10437
authors: Oreshkin, Carpov, Chapados, Bengio (Element AI and Mila; from the paper's first page)
year: 2019
status: body-read
field: time series forecasting
---

# N-BEATS 2019

> **In one line for a 12-year-old:** A stack of plain neural-network blocks takes a window of the past numbers, each block explains a part of them and passes on what is left over, and the parts they predict are added up to forecast the future. It beat the best classic forecasting methods in a big forecasting contest.

Paper: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", arXiv 1905.10437. First posted on 24 May 2019 (arXiv listing). I read version 4 of 20 February 2020, which carries the header "Published as a conference paper at ICLR 2020" (paper's first page). It is about univariate [[Time Series]] point forecasts.

## The problem
In forecasting, deep learning had a weak record against classic statistical methods. The paper's example: in the M4 competition, the six "pure" machine-learning entries finished 23rd, 37th, 38th, 48th, 54th and 57th out of 60, and most of the best entries were ensembles of classic statistical methods. The M4 winner was a hybrid of a dilated [[LSTM]] stack with a [[Exponential Smoothing]] (Holt-Winters) model, so it leaned heavily on a time-series component. The authors ask: can a purely deep-learning model do better, and can it be made interpretable like a trend-and-season decomposition? (paper)

## What people did before
- **Statistical methods:** exponential smoothing and its variants, Theta (winner of the earlier M3 contest), [[ARIMA]], and combinations of them. The paper calls exponential smoothing a default choice in industry (paper).
- **ML combined with statistics:** the second M4 entry computed the outputs of several statistical methods and combined them with [[Gradient Boosting]] (paper).
- **Deep learning:** mostly [[Recurrent Neural Network]] variants, strongest in electricity-load forecasting, plus work combining recurrent nets with dilation, residual links and attention (paper).
- **The M4 winner (Smyl):** a hybrid with a Holt-Winters-style model, hand-adjusted for each forecast horizon, which the authors call hard to generalise (paper).

## The idea
1. **A simple block.** Each block takes a window of past values (its input) and passes it through four fully-connected layers with [[ReLU]] ([[Feed-Forward Network]]). Two linear outputs give "expansion coefficients": one set for the **forecast** and one for the **backcast**, which is the block's own best reconstruction of its input (paper).
2. **Doubly residual stacking.** The next block receives the input **minus** the previous block's backcast, so each block removes the part of the signal it can already explain. The final forecast is the **sum** of all the blocks' partial forecasts (paper). This extends the idea of [[Residual Connection]] ([[ResNet 2015]]) to two branches.
3. **Generic version.** The step from coefficients to output is a learned linear layer, so no time-series knowledge is built in: no feature engineering and no input scaling (paper). The partial forecasts are then not interpretable.
4. **Interpretable version.** Two stacks with a fixed shape for the output: a **trend** stack whose output is a low-degree polynomial in time, and a **seasonality** stack whose output is a sum of sines and cosines (a Fourier series). The trend is removed from the input before the seasonality stack sees it, and the two partial forecasts can be read separately, like a decomposition ([[Trend and Seasonality]]) (paper).
5. **Ensemble.** The reported results use 180 models: trained on three different loss functions (sMAPE, MASE and a MAPE variant, see [[Forecast Metrics]]), with six input-window lengths (2 to 7 times the forecast horizon), and different random starts. The ensemble takes the **median**. The authors found ensembling a far stronger regulariser than dropout or L2 penalties ([[Ensemble Methods]]) (paper).
6. **Training.** One model per forecast horizon, shared across all the series of a dataset; batches of 1,024 randomly chosen series and forecast points; [[Adam Optimizer]] with learning rate 0.001; early stopping on a validation set (paper).

## What they found
Each number is marked "abstract" or "paper" (read in the body text, as extracted from the PDF).
- **M4 (100,000 series), sMAPE and OWA (lower is better; paper, Table 1):** N-BEATS-G (generic) 11.168 and 0.797; N-BEATS-I (interpretable) 11.174 and 0.798; N-BEATS-I+G (ensemble of both) 11.135 and 0.795. Against them: the M4 winner (DL and statistics hybrid) 11.374 and 0.821; the second-place ML and statistics combination 11.720 and 0.838; the third entry (ProLogistica) 11.845 and 0.841; the best purely statistical entry 11.986 and 0.861; the best pure ML entry 12.894 and 0.915. OWA is an M4 score normalised so that a seasonal naive forecast scores 1.0 (paper).
- **Stated improvements:** about 11% over a statistical benchmark, 7% over the best statistical entry and 3% over the M4 winner (abstract and paper). By my arithmetic from Table 1, the OWA of 0.795 against the winner's 0.821 is 3.2% lower, and the sMAPE of 11.135 against 11.374 is 2.1% lower.
- **M3 (3,003 series), sMAPE:** N-BEATS-G 12.47, N-BEATS-I 12.43, N-BEATS-I+G 12.37, against the best earlier entry EXP at 12.71 and the Theta method at 13.01 (paper).
- **TOURISM (1,311 series), MAPE:** N-BEATS-G 18.47, -I 18.97 and -I+G 18.52, against the best other entry at 19.35 and the ETS benchmark at 20.88 (paper).
- **One architecture for everything:** the same architecture and hyperparameters were used across horizons and datasets, from 48,000 monthly series in M4 down to 174 series in a small M3 subset (paper).
- **Interpretability costs nothing here:** the interpretable version was within about 0.001 of the generic one on the M4 OWA (0.798 against 0.797). Its trend output was monotonic and slowly moving and its seasonality output regular and cyclical, while the generic version's stack outputs were arbitrary (paper, Figure 2 caption and text).
- **Gap versus the winner:** the OWA gap between N-BEATS and the M4 winner (0.026) is larger than the gap between the winner and the second entry (0.017), as the authors point out (paper).

## Why it mattered
It was presented as the first demonstration, in the authors' words, that pure deep learning with no time-series-specific components can beat well-established statistical methods on M3, M4 and TOURISM. That strengthened the case for deep learning in forecasting, and it showed that a deep model can also be built to give a trend-and-season style explanation (paper). From memory (not checked in this paper): the architecture was influential and was extended by later models, and it is a standard baseline in deep forecasting research. The later paper [[DLinear 2022]] found that on long-horizon benchmarks very simple models can beat complex ones, which is a useful counterweight.

## Limits and open questions
- **Univariate point forecasts only:** one series at a time, with no outside variables and no forecast intervals (paper, problem statement).
- **It learns from many series:** the strong results are from datasets with thousands to 100,000 series. I did not check how it behaves on a single short series such as the airline data in [[Forecasting Shoot-out]]; the paper does not claim it.
- **Costly to match:** the reported results use a 180-model ensemble, and the authors report 30 minutes to 2 hours of GPU time to train one ensemble member on all of M4 (paper). One model is trained per forecast horizon.
- **Benchmarks are competitions:** the evidence is from three datasets and their standard metrics. The authors' own explanation of why it works well (a form of meta-learning, a fast "inner" adaptation inside the stack) is called a speculation that needs deeper investigation (paper, conclusion).
- **Window and metric choices matter:** results depend on the input window lengths and on training with several metrics, which the ensemble uses for diversity (paper).

## Concepts used
- [[Time Series]] · [[Trend and Seasonality]] · [[Exponential Smoothing]] · [[ARIMA]] · [[Forecast Metrics]] · [[Residual Connection]] · [[Feed-Forward Network]] · [[ReLU]] · [[Neural Network]] · [[Ensemble Methods]] · [[Gradient Boosting]] · [[LSTM]] · [[Recurrent Neural Network]] · [[Adam Optimizer]] · [[Regularization]] · [[Dropout]] · [[Backpropagation]]

## Reading trail
- **Came from:** [[ResNet 2015]] (residual connections), and the classic methods in [[Exponential Smoothing]] and [[ARIMA]] that it was measured against.
- **Led to:** no paper in this vault follows from it directly. [[DLinear 2022]] is a later, related question (are complex deep forecasters needed?) on different benchmarks.

## How much I read
I downloaded the full paper (31 pages) and read the extracted text of: the abstract, the introduction, the problem statement, the method (sections 3.1 to 3.4), related work, the experiments (section 5, including Table 1), the interpretability results, the meta-learning discussion and the conclusion.

**Not read:** the appendices (dataset statistics, ablation studies on ensemble size and on the doubly residual design, the detailed results by data frequency, and the hyperparameters). Figures were read only through their captions and the text around them. The table values come from text extracted from the PDF in column order; I matched each to its row. I did not re-run any experiment. The paper's own claims such as "first to demonstrate" are reported as claims. The note was not opened in Obsidian.

## Sources
- Oreshkin, Carpov, Chapados & Bengio, "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", arXiv 1905.10437 (v1 24 May 2019, v4 20 February 2020): https://arxiv.org/abs/1905.10437

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS (Oreshkin, Carpov, Chapados, Bengio, 2019, arXiv 1905.10437): what problem did it tackle? (paper)::Deep learning had a weak record against statistical forecasting. In M4 the six pure ML entries finished 23rd to 57th of 60. Can a purely deep model beat the statistical and hybrid methods, and can it be made interpretable?

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: how does one block work? (paper)::A window of past values goes through four fully-connected ReLU layers. Two linear outputs give coefficients for a forecast (forward) and a backcast (the block's reconstruction of its own input).

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: what is doubly residual stacking? (paper)::Each block's input is the previous input minus the previous block's backcast, so a block removes what is already explained. The final forecast is the sum of all the blocks' partial forecasts.

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: what is the difference between the generic and interpretable versions? (paper)::Generic uses a learned linear layer for the output and has no time-series knowledge, so its partial forecasts are not interpretable. Interpretable uses a trend stack (low-degree polynomial output) followed by a seasonality stack (Fourier series output), so the two parts can be read separately.

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: what was the ensemble? (paper)::180 models, trained with three losses (sMAPE, MASE, a MAPE variant), six input-window lengths (2 to 7 times the horizon) and different random starts. The median is taken. The authors found ensembling a stronger regulariser than dropout or L2.

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: what were the M4 results against the winner? (paper, my arithmetic where marked)::Ensemble N-BEATS-I+G: sMAPE 11.135, OWA 0.795. M4 winner (DL and statistics hybrid): 11.374 and 0.821. Statistical best entry: 11.986 and 0.861. Stated gains: 11% over the statistical benchmark, 7% over the best statistical entry, 3% over the winner. My arithmetic: OWA 3.2% and sMAPE 2.1% lower than the winner.

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: what were the M3 and TOURISM results? (paper)::M3 sMAPE: N-BEATS-I+G 12.37 against EXP 12.71 and Theta 13.01. TOURISM MAPE: N-BEATS-G 18.47 against the best other entry 19.35 and ETS 20.88.

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: did interpretability cost accuracy? (paper)::No. On M4 the interpretable version had OWA 0.798 against 0.797 for the generic one. Its trend output was monotonic and slowly moving, and its seasonality output regular and cyclical.

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: what are its limits? (paper)::Univariate point forecasts only (no outside variables, no intervals). Strong results come from datasets with thousands to 100,000 series, not tested on one short series. The reported results use a 180-model ensemble and one model per horizon, and 30 minutes to 2 hours of GPU time per ensemble member on M4. Evidence is from three competition datasets.

#flashcards/research/concept #flashcards/topic/research-time-series N-BEATS: what is its place in the reading trail?::It came from ResNet (residual connections) and the classic exponential smoothing and ARIMA methods it was compared with. No paper note follows from it directly. DLinear 2022 is a later related question on different benchmarks.

#flashcards/research/concept #flashcards/topic/research-time-series How much of the N-BEATS paper did the note read, and what was not read?::The abstract, introduction, problem statement, method, related work, experiments (Table 1), interpretability results, discussion and conclusion. Not read: all appendices (datasets, ablations, results by frequency, hyperparameters). Figures only through captions. No experiment re-run.
