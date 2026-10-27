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

## 7. Within-session binary verification

The subsequent [signal-baseline and reliability report](BASELINE_ROBUSTNESS_REPORT.md) adds raw-signal baseline controls, conditional participant uncertainty, validation-threshold sensitivity, and a matched seen/unseen-impostor comparison. The original results below are retained as originally reported; the follow-up explains the count and sampling differences in the initial unseen-impostor comparison.

Each participant was evaluated as the authorized user against the other 26 identities. The primary protocol uses the same chronological 20 / 7 / 7 split. A StandardScaler and class-balanced logistic regression are fitted on training observations only. The user-specific acceptance threshold is selected from validation scores at the nearest finite empirical operating point to the interpolated EER crossing, then frozen for test evaluation.

Values below are macro averages, giving every authorized user equal weight. Test EER is calculated from continuous test scores; test FAR and FRR use the validation-selected threshold.

| Experiment | ROC-AUC | EER | FAR | FRR |
|---|---:|---:|---:|---:|
| All six features | **0.855** | 0.178 | 0.157 | 0.217 |
| Means only | 0.838 | **0.167** | 0.174 | 0.222 |
| Std/range only (both means excluded) | 0.672 | 0.361 | 0.297 | 0.429 |
| Unseen impostors, all six (10 identity partitions) | 0.822 | 0.200 | 0.180 | 0.251 |

- **Not an acceptable security operating point.** With all six features, 15.7% of known-impostor requests were falsely accepted while 21.7% of genuine requests were rejected. Always rejecting would achieve 96.3% conventional accuracy in this imbalanced task, so raw accuracy is not a useful headline metric.
- **Mean-dependent separation dominates.** Means-only ROC-AUC is close to the all-six result, while excluding both means reduces ROC-AUC to 0.672 and raises EER to 0.361. The slightly lower means-only EER than all-six EER should not be interpreted as superiority: AUC is lower, thresholds differ by user, and each user has only seven genuine test windows.
- **Chronology matters, but less than feature choice here.** A secondary fixed-seed random-window verification split produced all-six ROC-AUC 0.882 versus 0.855 chronologically. This is an optimism diagnostic on the same recordings, not an independent replication.
- **Unseen attacker identities reduce performance modestly.** Holding impostor identities out of fitting lowers all-six ROC-AUC by 0.033 and raises EER by 0.022. It tests generalization to different impostors, not to a new session from the genuine user.
- **Recording artifacts remain a credible explanation.** Median per-participant linear-trend R² is 0.91 for A_mean and 0.93 for B_mean, versus 0.02–0.03 for the spread features. Removing means is only a feature ablation; it is not equivalent to correcting raw-electrode DC offsets.

These results support within-session verification signal, but do not establish a persistent biometric. The decisive follow-up is at least three independently collected sessions per participant, with electrodes removed and reapplied: enrollment on one session, threshold selection on a second, and a single untouched test on a third, including impostor identities absent from model fitting.
