# Research Summary: Within-Session EOG Participant Identification

These are aggregate results only. They are reproduced by the scripts in this repository on the dataset copy identified in `data_checksums.sha256`. Participant-level outputs are generated locally under `results/` and are not part of this repository.

**Setup:**
- 27 participants, one ~69-s recording each, 250 Hz, two channels (A horizontal, B vertical).
- 34 non-overlapping 2-s windows per participant, 918 windows in total.
- Six features: mean, std and range of each channel.
- Per-participant 20 / 7 / 7 train / validation / test split.
- StandardScaler fit on training windows only.
- Chance accuracy is 1/27 ≈ 0.037. With 7 test windows per participant, one test window is ≈ 0.5 percentage points.

## 1. Baseline: logistic regression, all six features

| Split protocol | Train acc | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|
| Chronological holdout (primary) | 0.732 | 0.593 | **0.545** | 0.492 |
| Random within-session split, seed 0 | 0.657 | 0.635 | 0.582 | 0.574 |

## 2. Feature ablation

**Chronological holdout**, identical split and model; only the feature columns change:

| Feature set | Train acc | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|
| All 6 | 0.732 | 0.593 | **0.545** | 0.492 |
| Means only (A_mean, B_mean) | 0.693 | 0.439 | **0.349** | 0.253 |
| Std/range only (A_std, A_range, B_std, B_range) | 0.193 | 0.116 | **0.148** | 0.127 |

**Random within-session split, seeds 0–99.** Each seed's split is shared by all three feature sets, so comparisons are paired. These are 100 repartitions of the same 27 recordings, not independent replications.

| Feature set | Test acc: mean (SD) | Median | 5th–95th pct | Min–max | Macro-F1 mean (SD) | Seeds above chronological |
|---|---|---|---|---|---|---|
| All 6 | 0.635 (0.034) | 0.635 | 0.582–0.688 | 0.550–0.720 | 0.620 (0.035) | 100/100 |
| Means only | 0.617 (0.028) | 0.619 | 0.576–0.667 | 0.545–0.672 | 0.583 (0.032) | 100/100 |
| Std/range only | 0.166 (0.021) | 0.164 | 0.137–0.201 | 0.116–0.238 | 0.136 (0.019) | 86/100 |

- **Random split vs chronological holdout:** the random split gave more optimistic results.
  - Means only: +0.27
  - All 6: +0.09
  - Std/range only: +0.02
- **Paired comparisons on the same random splits:**
  - All 6 beats std/range only by +0.469 on average, in 100/100 seeds.
  - All 6 beats means only by just +0.018 on average.

## 3. Within-recording drift (descriptive)

An ordinary least-squares line was fit to each participant × feature over window index. This is a simple description; it does not claim the drift is linear.

| Feature group | Median R² of linear trend | Median within- / between-participant SD |
|---|---|---|
| A_mean, B_mean | 0.91, 0.93 | 0.11, 0.14 |
| Std and range features | 0.02–0.04 | 0.86–1.04 |

- **Mean features:** they change smoothly over each recording, but stay small relative to the large differences between recordings. B_mean increased within 20 of 27 recordings.
- **Std/range features:** they show essentially no trend. Their window-to-window variation is about as large as their between-participant spread.
- **Spike caveat:** isolated spike windows can tilt the OLS slopes of std/range features, so a spike-robust change measure is also computed.

## 4. Logistic-regression coefficients (six-feature chronological model)

Coefficient magnitude describes how strongly this fitted linear classifier uses a standardized feature. It is **not** a measure of causal or physiological importance.

- **Ranking:** by mean absolute coefficient across the 27 classes, the order is B_mean (1.99), A_mean (1.87), B_std (0.62), B_range (0.61), A_range (0.55), A_std (0.42).
- **Share on the means:** the two mean features account for 64% of the total absolute coefficient mass. Per class, the median share is 62% (range 27–90%).
- **Extreme offsets:** the classes whose recordings have the most extreme DC offsets received the largest mean-feature coefficients.

## 5. Model comparison

Fixed, documented parameters with **no tuning**; the test set is not used for any selection. Ranges are chronological test accuracy across the six models: LR, LDA, linear SVM, RBF SVM, random forest and kNN.

| Feature set | Chronological test acc (range across models) | Random split > chronological? |
|---|---|---|
| All 6 | 0.513–0.571 | Yes, for all 6 models |
| Means only | 0.349–0.534 | Yes, for all 6 models |
| Std/range only | 0.122–0.190 | Yes, for all 6 models |

- **Largest optimism gaps:** they come from flexible, local models. Random forest is +0.30 with all six features and +0.46 with means only.
- **No winner is chosen:** differences of a few test windows between models are not meaningful.

## 6. Interpretation and limits

- **Mean features dominate.** With these six features, most of the within-session participant-discriminative signal comes from the per-window mean features. Those features behave like recording-level DC offsets with slow drift.
- **Without the means, accuracy collapses.** Removing them lowers chronological accuracy from 0.545 to 0.12–0.19, depending on the model. It also nearly removes the random-vs-chronological gap.
- **Offset-insensitive signal is weak.** The std/range features carry some participant-discriminative signal (about 4–5× chance), but it is weak and noisy at the window level.

What these results do **not** establish:
- **No stable identity.** They do not show that any feature reflects identity that is stable across sessions, because there is only one session per participant.
- **Offset origin unknown.** They do not show whether the offsets are physiological or come from equipment or session factors.
- **The optimism is not leakage.** No window, scaler statistic or label crosses splits; the gap reflects temporal dependence within each recording.
- **Crude features only.** They do not show that offset-insensitive EOG structure is weak in general. Only four crude statistics were tested.

**Suggested next steps:**
- **Cross-session evaluation** (at least two sessions per participant) is required for any identity claim.
- **Within this dataset,** test offset-invariant features (per-window detrending, event-based saccade/blink/fixation features, spectral features) under chronological holdout. Any normalization must be computed per window or from training data only; per-participant normalization would leak the label.
