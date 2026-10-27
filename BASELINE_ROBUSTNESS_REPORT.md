# EOG Signal Baselines and Authentication Reliability

## Executive summary

The within-session classifiers depend substantially on baseline-sensitive features. Removing the local window mean gives ROC-AUC 0.672, compared with 0.855 for the original six features. This is consistent with recording-dependent electrical levels contributing to separation, but does not identify their cause.

The additional all-six versus means-only ROC-AUC is only 0.016; its conditional 95% identity-reweighting interval is [-0.026, 0.066]. The much larger all-six versus std/range difference is 0.183, with interval [0.125, 0.242]. These quantify uncertainty in the fitted experiment, not reliability across days or independent training cohorts.

## 1. Research question and audit of existing work

To what extent does within-session EOG authentication depend on baseline-sensitive features, and how robust are the findings statistically?

The existing feature extraction computes mean, population standard deviation, and range independently on each channel of each non-overlapping 500-sample window. No raw-signal baseline subtraction or filter is applied. StandardScaler then subtracts a pooled training-feature mean and divides by pooled training-feature SD; this rescales the predictors but retains differences between recording baselines. All transformations and logistic regression parameters in this follow-up were specified before executing this follow-up; the earlier test set has already been inspected, so all new comparisons are exploratory and need independent confirmation.

Existing drift analysis and coefficient inspection already suggest mean dependence. This follow-up adds raw-signal controls, identity-level uncertainty, validation deletion sensitivity, and a matched attacker comparison. No original output was overwritten.

## 2. Dataset and methods

All 27 raw recordings are accessible and match all 27 recorded SHA-256 checksums. Each contributes 34 chronological two-second windows at 250 Hz: 918 windows in total, with 20 enrollment/training, 7 validation and 7 test windows per participant. Raw signals are read only. Signal magnitudes are reported in stored units; no independent voltage calibration or electrode-placement measurement was available to this analysis. One recording per participant is treated as one session; no independent sessions are inferred from the windows.

Each user retains the existing class-balanced logistic regression, C=1, train-only StandardScaler, validation-only empirical EER threshold, and frozen test evaluation within each specified comparison. EER is interpolated on score ROC points; FAR/FRR use the frozen empirical validation threshold, and therefore need not equal EER or each other. Baseline plots using all windows are descriptive only; nearest-centroid separation uses enrollment windows only and never changes a model or threshold.

## 3. Existing authentication results

| Original experiment | ROC-AUC | EER | FAR | FRR |
|---|---:|---:|---:|---:|
| All six features | 0.855 | 0.178 | 0.157 | 0.217 |
| Means only | 0.838 | 0.167 | 0.174 | 0.222 |
| Std/range only | 0.672 | 0.361 | 0.297 | 0.429 |
| Original unseen-impostor protocol | 0.822 | 0.200 | 0.180 | 0.251 |

Random-window all-six ROC-AUC was 0.882 versus 0.855 chronologically. These earlier values are preserved. In the original unseen protocol, impostors contribute all 34 windows per assigned identity, whereas the primary known protocol uses chronological 20/7/7 windows from every identity. Its difference combines attacker exposure, counts, and temporal sampling; it is not an isolated unseen-attacker effect.

## 4. What the signal means show

| Feature | Between-recording SD | Median within-recording SD | Median within/between SD | Median trend R-squared | Positive slopes |
|---|---:|---:|---:|---:|---:|
| A_mean | 3.177 | 0.362 | 0.114 | 0.913 | 11/27 |
| B_mean | 3.500 | 0.502 | 0.144 | 0.932 | 20/27 |

Between-recording SD is the sample SD of the 27 recording means; within SD is the sample SD across a recording's 34 window means. They describe electrical levels, not independent physiological identities. Small within/between ratios show that offsets differ more across recordings than across windows of a recording, even though slow drift is present.

The enrollment centroid's nearest-neighbor distance has exploratory Spearman correlation 0.446 with means-only test ROC-AUC. Participants with distinct means can be easy to separate, but distance alone does not ensure success: a one-versus-rest linear decision boundary can struggle with interior clusters, and drift can move genuine windows across that boundary. These are possible explanations, not measured causes of individual errors. The local participant table identifies isolated and overlapping recordings and their authentication performance; participant IDs and exact electrical levels remain in the ignored results directory. The scatterplot shows every participant; annotations identify only the smallest/largest nearest-centroid distances and the lowest means-only AUC, selected by these fixed descriptive criteria.

![Per-participant distributions](results/baseline_robustness/baseline_distributions.png)

![Raw and enrollment-centered trajectories](results/baseline_robustness/baseline_trajectories.png)

![Within versus between variability](results/baseline_robustness/baseline_variability.png)

![Baseline separation and authentication performance](results/baseline_robustness/baseline_performance_relationship.png)

## 5. Controlled baseline and offset experiments

No transformation estimates a baseline across a participant's whole recording. Every window is processed separately as a two-second authentication trial with the same transformation in training, validation, and testing. These trials remain temporally correlated:

- **Window centering:** z[t] = x[t] - mean(x within this trial). This requires the completed two-second input, as does the original feature extraction; it learns no population or session parameter from held-out data. It removes every constant additive shift and also removes any physiological information in absolute average level. Its output mean is exactly zero by construction. The four std/range features are mathematically unchanged, so equality with std/range-only is an algebraic control, not independent evidence.
- **First-sample reference:** z[t] = x[t] - x[0]. This is causal and uses a reference available at trial start. It removes a constant additive offset while retaining movements relative to the first sample. It can inject sensitivity to noise or eye state at that first sample and does not remove drift within the trial.
- **Causal DC blocker:** z[0]=0; z[t]=x[t]-x[t-1]+alpha*z[t-1], alpha=exp(-2*pi*0.5/250). The fixed 0.5-Hz setting is a sensitivity probe, not an optimized physiological cutoff. State resets each trial, and the output uses only current/past samples. It removes a constant offset, attenuates slower components, and can also remove physiological low-frequency/eye-position information. Reset behavior can affect dynamics. No filter setting was chosen from test performance.

All transformed six-feature matrices use the same scaler/model procedure. Mathematically zero centered means are forced to numerical zero to prevent the scaler from amplifying cancellation noise. Additive-shift invariance is tested on synthetic signals. These transformations do not remove gain, impedance, morphology, noise, or other acquisition effects; offset invariance does not imply artifact freedom.

| Variant | ROC-AUC | 95% identity sensitivity interval | EER | 95% identity sensitivity interval | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|
| Original six | 0.855 | [0.808, 0.896] | 0.178 | [0.135, 0.228] | 0.157 | 0.217 |
| Means only | 0.838 | [0.790, 0.879] | 0.167 | [0.125, 0.214] | 0.174 | 0.222 |
| Std/range | 0.672 | [0.596, 0.744] | 0.361 | [0.296, 0.428] | 0.297 | 0.429 |
| Window centered | 0.672 | [0.596, 0.744] | 0.361 | [0.296, 0.428] | 0.297 | 0.429 |
| First-sample reference | 0.681 | [0.604, 0.755] | 0.365 | [0.293, 0.431] | 0.280 | 0.407 |
| Causal DC blocker | 0.698 | [0.631, 0.762] | 0.338 | [0.282, 0.397] | 0.279 | 0.423 |

![Transformations and uncertainty](results/baseline_robustness/baseline_transformation_comparison.png)

## 6. Statistical reliability

Two complementary percentile resampling analyses use 5000 fixed-seed draws. The authorized-task bootstrap draws 27 complete user tasks with replacement and retains every correlated window. It conditions on the existing shared impostor panel. The crossed identity bootstrap uses one common identity count vector to reweight the same participants in both authorized-task and impostor roles, recomputes each frozen-score ROC-AUC/EER, and macro-averages with those same target weights. Thus all windows from one identity move together, and an identity reused as an impostor across tasks has a shared weight. Tied scores are grouped in the weighted ROC calculation.

The table shows crossed-identity 2.5th/97.5th percentiles. Both interval types are saved in JSON. These are approximate, conditional sensitivity intervals: neither refits models or scalers, neither reselects thresholds, and neither captures dependence from shared training data, unseen sessions, new acquisition conditions, or genuine-window sampling variation within each fixed recording. They should not be read as calibrated population confidence intervals for a deployable biometric. The 918 windows and the 270 repeated user/partition tasks are never treated as independent subjects.

| Paired ROC-AUC contrast | Mean difference | 95% crossed-identity interval | Users positive / negative / tied |
|---|---:|---:|---:|
| original minus means only | +0.016 | [-0.026, 0.066] | 12/10/5 |
| original minus std range only | +0.183 | [0.125, 0.242] | 25/2/0 |
| means only minus std range only | +0.167 | [0.074, 0.255] | 20/7/0 |
| first sample reference minus original | -0.174 | [-0.235, -0.114] | 3/24/0 |
| causal dc blocker minus original | -0.157 | [-0.206, -0.109] | 3/24/0 |

The paired contrasts reuse identical resampling counts and test requests. An interval including zero does not establish an advantage; an interval excluding zero supports a conditional difference in these recordings, not its physiological cause. Multiple exploratory comparisons are reported without a confirmatory significance claim. Per-user all-six ROC-AUC spans 0.557-1.000; EER spans 0.000-0.440. One genuine error changes a user's FRR by 1/7 = 14.3 percentage points; its small window-level denominator is a major precision limit.

### Threshold sensitivity

With each fitted model fixed, thresholds are separately recalculated after deleting one of the seven genuine validation windows, or one complete validation impostor identity. The resulting test FAR/FRR are sensitivity diagnostics, never used to choose a preferred threshold or replace the primary results. This jackknife-like deletion analysis is not a confidence interval and assumes no independence of neighboring windows.

For all six features, deleting one genuine validation window gives a median per-user test FAR range of 0.016 and FRR range of 0.000; the largest ranges are 0.143 and 0.714. In 12/27 users, at least one such deletion changes test FRR relative to the original threshold. Exact deletion outcomes remain local.

![Threshold sensitivity](results/baseline_robustness/baseline_threshold_sensitivity.png)

## 7. Seen and unseen impostors under matched conditions

For each target and each of ten fixed partitions, both arms use 16 training impostor identities with 20 early windows each, 5 validation identities with 7 middle windows each, and exactly the same 5 test identities with 7 late windows each. Genuine counts remain 20/7/7. Validation identities are absent from both training arms. In the seen arm, five of the 16 training identities are the test identities; the unseen arm replaces those five with other identities. Training size, imbalance, validation requests and test requests match; training composition necessarily changes to manipulate test-identity exposure. Each arm selects its threshold using its own validation scores.

| Features | Seen test ROC-AUC | Unseen test ROC-AUC | Unseen minus seen | 95% paired task interval |
|---|---:|---:|---:|---:|
| Original six | 0.856 | 0.812 | -0.044 | [-0.060, -0.029] |
| Means only | 0.838 | 0.797 | -0.041 | [-0.058, -0.024] |
| Std/range | 0.680 | 0.667 | -0.012 | [-0.018, -0.007] |

Repeated partitions are averaged within each authorized user before participant resampling. Intervals remain conditional on the cohort/shared training and test panels. Repeats quantify partition sensitivity rather than independent replications. These matched results better isolate exposure to test impostor identities than the earlier unmatched comparison. They still do not test a later session from the genuine user.

![Matched attacker comparison](results/baseline_robustness/baseline_matched_impostors.png)

## 8. Interpretation and alternative explanations

**Observation:** absolute window means differ across recordings; slow drift is strong in mean features; mean-based separation survives within a recording; the offset-invariant transformations have the measured performance above.

**Interpretation:** this pattern suggests baseline dependence and is consistent with electrode placement, electrode-skin contact, amplifier/reference offsets, or other recording-specific characteristics contributing to the fitted separation. None of those mechanisms was measured or manipulated here, so this analysis cannot attribute the effect to electrode placement or prove that artifacts explain most performance.

**Alternative explanations:** stable physiological differences could influence recorded levels or amplitudes; eye position, task behavior, anatomy and physiological low-frequency signals could also contribute. Baseline removal deletes information indiscriminately, including any useful physiological level information. Conversely, residual performance after removing additive offsets could still arise from acquisition gain/noise or session-specific behavior. The observations do not separate these explanations because identity and session are confounded.

## 9. What remains unknown and the next experiment

The original ROC-AUC point estimates are reproducible descriptions of these recordings, with substantial user variation and conditional uncertainty. They do not establish persistent biometric identity, generalization across days, reliable security operating points, or artifact causation. New transformations were evaluated after observing the original benchmark, so independent confirmation is required rather than choosing a winning transformation from this test set.

The single most valuable experiment is a prospective, crossed multi-session collection: at least three days per participant, electrodes removed and reapplied between sessions, with placement/reference/contact quality and amplifier calibration recorded. Include a controlled reapplication within a day to distinguish reapplication from day effects; randomize placement/operator/order where practical and hold task/gaze instructions consistent. Enroll on session one, select preprocessing and thresholds using session two, lock the protocol, and evaluate once on session three with impostor identities absent from fitting. Prespecify a target FAR and acquire enough independent impostor attempts and sessions to estimate it. This tests whether identity separation persists while electrical acquisition conditions change.

## 10. Reproduction and evidence

Run `python run_baseline_robustness.py` after setting `EOG_RAW_DIR` as documented in README. Run `python -m unittest test_baseline_robustness.py` for data-free mathematical, causal and uncertainty tests. A nonempty output directory is rejected; use `--output results/baseline_robustness_replication` for a second preserved run. CSVs retain per-user metrics, baseline summaries, threshold deletions, matched assignments and comparisons. JSON retains both uncertainty methods, exact configuration, source checksums, and dependency versions. Previous authentication outputs remain unchanged; participant-level tables and all new figures stay in ignored local results.
