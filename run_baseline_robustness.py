"""Run the baseline-sensitivity follow-up without changing earlier outputs.

Set EOG_RAW_DIR if recordings live outside the repository. A nonempty output
directory is rejected; use --output results/baseline_robustness_replication to
preserve an earlier run. All participant-level outputs remain under results/.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import authentication as auth
import baseline as lb
import baseline_robustness as br


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / 'results' / 'baseline_robustness'
LABELS = {'original': 'Original six', 'means_only': 'Means only',
          'std_range_only': 'Std/range', 'window_centered': 'Window centered',
          'first_sample_reference': 'First-sample reference',
          'causal_dc_blocker': 'Causal DC blocker'}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_variants(config):
    X, y, windows, starts, info = lb.build_dataset()
    lb.check_dataset(X, y)
    assert X.shape == (918, 6)
    assert all(r['n_windows'] == 34 for r in info)
    assert np.all(starts == windows * lb.WINDOW_SIZE)
    matrices = {'original': X,
                'means_only': lb.select_features(X, lb.FEATURE_SETS['means_only']),
                'std_range_only': lb.select_features(X, lb.FEATURE_SETS['std_range_only'])}
    transformed = {name: [] for name in config['variants'] if name not in matrices}
    for pid in lb.PARTICIPANT_IDS:
        a, b, _ = lb.load_participant_signals(pid)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        aw, bw, _ = lb.extract_windows(a, b)
        for aa, bb in zip(aw, bw):
            for name in transformed:
                transformed[name].append(br.transformed_features(
                    aa, bb, name, config['dc_blocker_cutoff_hz']))
    matrices.update({name: np.asarray(rows) for name, rows in transformed.items()})
    return matrices, y, windows, info


def verify_data():
    records = []
    for line in (HERE / 'data_checksums.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = Path(lb.RAW_DIR) / Path(relative).name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Dataset checksum mismatch: {path.name}')
        records.append({'file': path.name, 'sha256': actual})
    return records


def known_experiments(matrices, y, split, config, model_config):
    records, details = {}, {}
    for name in config['variants']:
        rr, dd = auth.run_known_impostor(matrices[name], y, split,
                                         lb.PARTICIPANT_IDS, config['seed'], model_config)
        for row in rr:
            dd[row['participant']]['threshold'] = row['threshold']
        records[name], details[name] = rr, dd
    # Centering preserves exactly the same four spread statistics. Guard
    # against accidental numerical residual means becoming standardized inputs.
    assert np.allclose(matrices['window_centered'][:, [1, 2, 4, 5]], matrices['std_range_only'])
    assert np.all(matrices['window_centered'][:, [0, 3]] == 0)
    for centered, spread in zip(records['window_centered'], records['std_range_only']):
        assert np.isclose(centered['roc_auc'], spread['roc_auc'], atol=1e-12)
    return records, details


def uncertainty(records, details, y, config):
    counts = br.bootstrap_counts(27, config['bootstrap_draws'], config['seed'])
    summary, samples = {}, {}
    for name in config['variants']:
        crossed = br.crossed_identity_bootstrap(details[name], y, lb.PARTICIPANT_IDS, counts)
        samples[name] = crossed
        summary[name] = {}
        for metric in ['roc_auc', 'eer', 'far', 'frr']:
            summary[name][metric] = {
                'estimate': float(np.mean([r[metric] for r in records[name]])),
                'authorized_task_interval': br.interval(br.cluster_bootstrap(records[name], counts, metric)),
                'crossed_identity_interval': br.interval(crossed[metric]),
            }
    paired = {}
    for a, b in [('original', 'means_only'), ('original', 'std_range_only'),
                 ('means_only', 'std_range_only'), ('first_sample_reference', 'original'),
                 ('causal_dc_blocker', 'original')]:
        delta = np.array([ra['roc_auc'] - rb['roc_auc'] for ra, rb in zip(records[a], records[b])])
        paired[f'{a}_minus_{b}'] = {
            'mean_auc_difference': float(delta.mean()),
            'median_auc_difference': float(np.median(delta)),
            'users_positive': int(np.sum(delta > 1e-12)),
            'users_negative': int(np.sum(delta < -1e-12)),
            'users_tied': int(np.sum(np.abs(delta) <= 1e-12)),
            'authorized_task_interval': br.interval(counts @ delta / 27),
            'crossed_identity_interval': br.interval(samples[a]['roc_auc'] - samples[b]['roc_auc']),
        }
    return summary, paired


def threshold_audit(details, y):
    rows, summary = [], []
    for name in ['original', 'means_only', 'std_range_only']:
        for pid, detail in details[name].items():
            rr = br.threshold_deletions(detail, y)
            rows.extend({'variant': name, 'participant': pid, **r} for r in rr)
            for kind in ['delete_one_genuine', 'delete_one_impostor_identity']:
                selected = [r for r in rr if r['deletion'] == kind]
                summary.append({'variant': name, 'participant': pid, 'deletion': kind,
                                'threshold_min': min(r['threshold'] for r in selected),
                                'threshold_max': max(r['threshold'] for r in selected),
                                'far_min': min(r['far'] for r in selected),
                                'far_max': max(r['far'] for r in selected),
                                'frr_min': min(r['frr'] for r in selected),
                                'frr_max': max(r['frr'] for r in selected)})
    return rows, summary


def matched_protocol(matrices, y, split, config, model_config):
    rows, assignments = [], []
    for repeat in range(config['matched_impostor_repeats']):
        for target in lb.PARTICIPANT_IDS:
            groups = auth.make_unseen_identity_split(target, lb.PARTICIPANT_IDS,
                                                     config['seed'] + repeat)
            for seen in [False, True]:
                indices = br.matched_impostor_indices(y, split, target, groups, seen)
                assert [len(indices[k]) for k in ['train', 'val', 'test']] == [340, 42, 42]
                for name in ['original', 'means_only', 'std_range_only']:
                    record, _ = auth.evaluate_target(matrices[name], y, target, indices,
                                                     config['seed'] + repeat, model_config)
                    rows.append({'variant': name, 'condition': 'seen' if seen else 'unseen',
                                 'repeat': repeat, **record})
                for partition, indices_part in indices.items():
                    assignments.extend({'participant': target, 'repeat': repeat,
                                        'condition': 'seen' if seen else 'unseen',
                                        'split': partition, 'impostor': int(pid)}
                                       for pid in np.unique(y[indices_part]) if pid != target)
    summary = {}
    counts = br.bootstrap_counts(27, config['bootstrap_draws'], config['seed'])
    for name in ['original', 'means_only', 'std_range_only']:
        summary[name] = {}
        user_values = {}
        for condition in ['seen', 'unseen']:
            rr = [r for r in rows if r['variant'] == name and r['condition'] == condition]
            summary[name][condition] = auth.aggregate(rr)
            user_values[condition] = np.array([
                np.mean([r['roc_auc'] for r in rr if r['participant'] == p]) for p in lb.PARTICIPANT_IDS])
            repeat_means = [np.mean([r['roc_auc'] for r in rr if r['repeat'] == k])
                            for k in range(config['matched_impostor_repeats'])]
            summary[name][condition]['repeat_auc_min'] = float(min(repeat_means))
            summary[name][condition]['repeat_auc_max'] = float(max(repeat_means))
        delta = user_values['unseen'] - user_values['seen']
        summary[name]['unseen_minus_seen_auc'] = float(delta.mean())
        summary[name]['paired_task_interval'] = br.interval(counts @ delta / 27)
    return rows, assignments, summary


def plot_results(out, matrices, y, windows, baseline_rows, records, stats, threshold_summary, matched):
    pids = lb.PARTICIPANT_IDS
    original = matrices['original']
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, column in zip(axes, [0, 3]):
        values = [original[y == p, column] for p in pids]
        ax.boxplot(values, tick_labels=[f'{p:02d}' for p in pids], showfliers=True)
        ax.set(ylabel=f'{lb.FEATURE_NAMES[column]} (stored signal units)', xlabel='Participant')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Mean EOG levels differ across recordings; units are not independently calibrated')
    save(fig, 'baseline_distributions.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, column in enumerate([0, 3]):
        for pid in pids:
            rows = np.where(y == pid)[0]
            values = original[rows, column]
            axes[j, 0].plot(windows[rows], values, alpha=.55, lw=.9)
            axes[j, 1].plot(windows[rows], values - values[:20].mean(), alpha=.55, lw=.9)
        for k in range(2):
            ax = axes[j, k]
            ax.axvline(19.5, color='gray', ls='--')
            ax.axvline(26.5, color='gray', ls='--')
            ax.set(title=lb.FEATURE_NAMES[column] + (' raw' if k == 0 else ' minus enrollment mean'),
                   xlabel='Chronological 2-second window', ylabel='Stored signal units')
            ax.grid(alpha=.2)
    save(fig, 'baseline_trajectories.png')

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, feature in enumerate(['A_mean', 'B_mean']):
        rr = [r for r in baseline_rows if r['feature'] == feature]
        ratios = [r['within_between_ratio'] for r in rr]
        ax.scatter(np.full(27, j), ratios, s=25, alpha=.65)
        ax.plot([j-.2, j+.2], [np.median(ratios)]*2, color='black')
    ax.axhline(1, color='gray', ls='--', label='Within SD equals between-recording SD')
    ax.set(xticks=[0, 1], xticklabels=['A_mean', 'B_mean'], ylabel='Within-recording SD / between-recording SD',
           title='Each dot retains one complete participant recording')
    ax.legend(fontsize=8)
    save(fig, 'baseline_variability.png')

    centers = np.array([original[y == p][:, [0, 3]].mean(axis=0) for p in pids])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = [r['roc_auc'] for r in records['means_only']]
    sc = axes[0].scatter(centers[:, 0], centers[:, 1], c=colors, vmin=min(colors), vmax=1,
                         cmap='viridis', s=65, edgecolor='black', linewidth=.4)
    nearest = [r['train_nearest_distance'] for r in baseline_rows if r['feature'] == 'A_mean']
    # All users are shown. Label only fixed descriptive extrema to keep the
    # densely overlapping central recordings legible; the CSV retains all IDs.
    annotate = {int(np.argmin(nearest)), int(np.argmax(nearest)), int(np.argmin(colors))}
    for index in sorted(annotate):
        axes[0].annotate(f'{pids[index]:02d}', centers[index], xytext=(4, 4),
                         textcoords='offset points', fontsize=8)
    axes[0].set(xlabel='Recording A_mean (stored units)', ylabel='Recording B_mean (stored units)',
                title='Means-only authentication performance')
    fig.colorbar(sc, ax=axes[0], label='Test ROC-AUC')
    axes[1].scatter(nearest, colors, color='#2474a6')
    axes[1].set(xlabel='Nearest enrollment centroid distance (standardized)', ylabel='Means-only test ROC-AUC',
                title='Separation does not guarantee linear verifiability')
    save(fig, 'baseline_performance_relationship.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, ['roc_auc', 'eer']):
        for i, name in enumerate(LABELS):
            m = stats[name][metric]
            ci = m['crossed_identity_interval']
            # Plot endpoints directly: a percentile interval need not straddle
            # the original estimate, so asymmetric errorbar lengths can be negative.
            ax.plot([ci['low'], ci['high']], [i, i], color='#2474a6', lw=2)
            ax.scatter(m['estimate'], i, color='black', s=25, zorder=3)
        ax.set(yticks=range(len(LABELS)), yticklabels=list(LABELS.values()),
               xlabel=metric.upper().replace('_', '-'), xlim=(0, 1))
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('95% identity-reweighting sensitivity intervals; frozen fitted models')
    save(fig, 'baseline_transformation_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rr = [r for r in threshold_summary if r['variant'] == 'original' and r['deletion'] == 'delete_one_genuine']
    for ax, metric in zip(axes, ['far', 'frr']):
        ax.vlines(pids, [r[f'{metric}_min'] for r in rr], [r[f'{metric}_max'] for r in rr],
                  color='#2474a6', lw=3, label='Range after deleting one genuine validation window')
        ax.scatter(pids, [r[metric] for r in records['original']], color='black', s=18,
                   label='Original threshold')
        ax.set(xlabel='Authorized participant', ylabel=f'Test {metric.upper()}', ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Validation threshold fragility: diagnostic only, no threshold replaced')
    save(fig, 'baseline_threshold_sensitivity.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metric in zip(axes, ['roc_auc', 'eer']):
        for i, name in enumerate(['original', 'means_only', 'std_range_only']):
            ax.plot([i-.12, i+.12], [matched[name]['seen'][metric], matched[name]['unseen'][metric]],
                    color='#777777')
            ax.scatter(i-.12, matched[name]['seen'][metric], marker='o', color='#2474a6',
                       label='Test identities seen in training' if i == 0 else None)
            ax.scatter(i+.12, matched[name]['unseen'][metric], marker='x', color='#b43c42',
                       label='Test identities unseen in training' if i == 0 else None)
        ax.set(xticks=range(3), xticklabels=['Original six', 'Means', 'Std/range'],
               ylabel=metric.upper(), ylim=(0, 1))
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Matched impostor comparison: equal counts and identical validation/test requests')
    save(fig, 'baseline_matched_impostors.png')


def fmt_ci(value):
    return f"[{value['low']:.3f}, {value['high']:.3f}]"


def make_report(result):
    stats, paired = result['uncertainty'], result['paired_comparisons']
    base, matched = result['baseline_summary'], result['matched_impostors']
    table = '\n'.join(
        f"| {LABELS[name]} | {stats[name]['roc_auc']['estimate']:.3f} | "
        f"{fmt_ci(stats[name]['roc_auc']['crossed_identity_interval'])} | "
        f"{stats[name]['eer']['estimate']:.3f} | {fmt_ci(stats[name]['eer']['crossed_identity_interval'])} | "
        f"{stats[name]['far']['estimate']:.3f} | {stats[name]['frr']['estimate']:.3f} |"
        for name in LABELS)
    comparisons = '\n'.join(
        f"| {name.replace('_', ' ')} | {r['mean_auc_difference']:+.3f} | "
        f"{fmt_ci(r['crossed_identity_interval'])} | {r['users_positive']}/{r['users_negative']}/{r['users_tied']} |"
        for name, r in paired.items())
    matched_table = '\n'.join(
        f"| {LABELS[name]} | {m['seen']['roc_auc']:.3f} | {m['unseen']['roc_auc']:.3f} | "
        f"{m['unseen_minus_seen_auc']:+.3f} | {fmt_ci(m['paired_task_interval'])} |"
        for name, m in matched.items())
    baseline_table = '\n'.join(
        f"| {feature} | {m['between_sd']:.3f} | {m['median_within_sd']:.3f} | "
        f"{m['median_within_between_ratio']:.3f} | {m['median_r2']:.3f} | {m['positive_slopes']}/27 |"
        for feature, m in base.items())
    threshold = result['threshold_summary']['original']
    return f"""# EOG Signal Baselines and Authentication Reliability

## Executive summary

The within-session classifiers depend substantially on baseline-sensitive features. Removing the local window mean gives ROC-AUC {stats['window_centered']['roc_auc']['estimate']:.3f}, compared with {stats['original']['roc_auc']['estimate']:.3f} for the original six features. This is consistent with recording-dependent electrical levels contributing to separation, but does not identify their cause.

The additional all-six versus means-only ROC-AUC is only {paired['original_minus_means_only']['mean_auc_difference']:.3f}; its conditional 95% identity-reweighting interval is {fmt_ci(paired['original_minus_means_only']['crossed_identity_interval'])}. The much larger all-six versus std/range difference is {paired['original_minus_std_range_only']['mean_auc_difference']:.3f}, with interval {fmt_ci(paired['original_minus_std_range_only']['crossed_identity_interval'])}. These quantify uncertainty in the fitted experiment, not reliability across days or independent training cohorts.

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
{baseline_table}

Between-recording SD is the sample SD of the 27 recording means; within SD is the sample SD across a recording's 34 window means. They describe electrical levels, not independent physiological identities. Small within/between ratios show that offsets differ more across recordings than across windows of a recording, even though slow drift is present.

The enrollment centroid's nearest-neighbor distance has exploratory Spearman correlation {result['separation_relationship']['spearman_rho']:.3f} with means-only test ROC-AUC. Participants with distinct means can be easy to separate, but distance alone does not ensure success: a one-versus-rest linear decision boundary can struggle with interior clusters, and drift can move genuine windows across that boundary. These are possible explanations, not measured causes of individual errors. The local participant table identifies isolated and overlapping recordings and their authentication performance; participant IDs and exact electrical levels remain in the ignored results directory. The scatterplot shows every participant; annotations identify only the smallest/largest nearest-centroid distances and the lowest means-only AUC, selected by these fixed descriptive criteria.

![Per-participant distributions](baseline_distributions.png)

![Raw and enrollment-centered trajectories](baseline_trajectories.png)

![Within versus between variability](baseline_variability.png)

![Baseline separation and authentication performance](baseline_performance_relationship.png)

## 5. Controlled baseline and offset experiments

No transformation estimates a baseline across a participant's whole recording. Every window is processed separately as a two-second authentication trial with the same transformation in training, validation, and testing. These trials remain temporally correlated:

- **Window centering:** z[t] = x[t] - mean(x within this trial). This requires the completed two-second input, as does the original feature extraction; it learns no population or session parameter from held-out data. It removes every constant additive shift and also removes any physiological information in absolute average level. Its output mean is exactly zero by construction. The four std/range features are mathematically unchanged, so equality with std/range-only is an algebraic control, not independent evidence.
- **First-sample reference:** z[t] = x[t] - x[0]. This is causal and uses a reference available at trial start. It removes a constant additive offset while retaining movements relative to the first sample. It can inject sensitivity to noise or eye state at that first sample and does not remove drift within the trial.
- **Causal DC blocker:** z[0]=0; z[t]=x[t]-x[t-1]+alpha*z[t-1], alpha=exp(-2*pi*0.5/250). The fixed 0.5-Hz setting is a sensitivity probe, not an optimized physiological cutoff. State resets each trial, and the output uses only current/past samples. It removes a constant offset, attenuates slower components, and can also remove physiological low-frequency/eye-position information. Reset behavior can affect dynamics. No filter setting was chosen from test performance.

All transformed six-feature matrices use the same scaler/model procedure. Mathematically zero centered means are forced to numerical zero to prevent the scaler from amplifying cancellation noise. Additive-shift invariance is tested on synthetic signals. These transformations do not remove gain, impedance, morphology, noise, or other acquisition effects; offset invariance does not imply artifact freedom.

| Variant | ROC-AUC | 95% identity sensitivity interval | EER | 95% identity sensitivity interval | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|
{table}

![Transformations and uncertainty](baseline_transformation_comparison.png)

## 6. Statistical reliability

Two complementary percentile resampling analyses use {result['config']['bootstrap_draws']} fixed-seed draws. The authorized-task bootstrap draws 27 complete user tasks with replacement and retains every correlated window. It conditions on the existing shared impostor panel. The crossed identity bootstrap uses one common identity count vector to reweight the same participants in both authorized-task and impostor roles, recomputes each frozen-score ROC-AUC/EER, and macro-averages with those same target weights. Thus all windows from one identity move together, and an identity reused as an impostor across tasks has a shared weight. Tied scores are grouped in the weighted ROC calculation.

The table shows crossed-identity 2.5th/97.5th percentiles. Both interval types are saved in JSON. These are approximate, conditional sensitivity intervals: neither refits models or scalers, neither reselects thresholds, and neither captures dependence from shared training data, unseen sessions, new acquisition conditions, or genuine-window sampling variation within each fixed recording. They should not be read as calibrated population confidence intervals for a deployable biometric. The 918 windows and the 270 repeated user/partition tasks are never treated as independent subjects.

| Paired ROC-AUC contrast | Mean difference | 95% crossed-identity interval | Users positive / negative / tied |
|---|---:|---:|---:|
{comparisons}

The paired contrasts reuse identical resampling counts and test requests. An interval including zero does not establish an advantage; an interval excluding zero supports a conditional difference in these recordings, not its physiological cause. Multiple exploratory comparisons are reported without a confirmatory significance claim. Per-user all-six ROC-AUC spans {result['participant_spread']['auc_min']:.3f}-{result['participant_spread']['auc_max']:.3f}; EER spans {result['participant_spread']['eer_min']:.3f}-{result['participant_spread']['eer_max']:.3f}. One genuine error changes a user's FRR by 1/7 = 14.3 percentage points; its small window-level denominator is a major precision limit.

### Threshold sensitivity

With each fitted model fixed, thresholds are separately recalculated after deleting one of the seven genuine validation windows, or one complete validation impostor identity. The resulting test FAR/FRR are sensitivity diagnostics, never used to choose a preferred threshold or replace the primary results. This jackknife-like deletion analysis is not a confidence interval and assumes no independence of neighboring windows.

For all six features, deleting one genuine validation window gives a median per-user test FAR range of {threshold['median_genuine_deletion_far_range']:.3f} and FRR range of {threshold['median_genuine_deletion_frr_range']:.3f}; the largest ranges are {threshold['max_genuine_deletion_far_range']:.3f} and {threshold['max_genuine_deletion_frr_range']:.3f}. In {threshold['users_with_frr_change']}/27 users, at least one such deletion changes test FRR relative to the original threshold. Exact deletion outcomes remain local.

![Threshold sensitivity](baseline_threshold_sensitivity.png)

## 7. Seen and unseen impostors under matched conditions

For each target and each of ten fixed partitions, both arms use 16 training impostor identities with 20 early windows each, 5 validation identities with 7 middle windows each, and exactly the same 5 test identities with 7 late windows each. Genuine counts remain 20/7/7. Validation identities are absent from both training arms. In the seen arm, five of the 16 training identities are the test identities; the unseen arm replaces those five with other identities. Training size, imbalance, validation requests and test requests match; training composition necessarily changes to manipulate test-identity exposure. Each arm selects its threshold using its own validation scores.

| Features | Seen test ROC-AUC | Unseen test ROC-AUC | Unseen minus seen | 95% paired task interval |
|---|---:|---:|---:|---:|
{matched_table}

Repeated partitions are averaged within each authorized user before participant resampling. Intervals remain conditional on the cohort/shared training and test panels. Repeats quantify partition sensitivity rather than independent replications. These matched results better isolate exposure to test impostor identities than the earlier unmatched comparison. They still do not test a later session from the genuine user.

![Matched attacker comparison](baseline_matched_impostors.png)

## 8. Interpretation and alternative explanations

**Observation:** absolute window means differ across recordings; slow drift is strong in mean features; mean-based separation survives within a recording; the offset-invariant transformations have the measured performance above.

**Interpretation:** this pattern suggests baseline dependence and is consistent with electrode placement, electrode-skin contact, amplifier/reference offsets, or other recording-specific characteristics contributing to the fitted separation. None of those mechanisms was measured or manipulated here, so this analysis cannot attribute the effect to electrode placement or prove that artifacts explain most performance.

**Alternative explanations:** stable physiological differences could influence recorded levels or amplitudes; eye position, task behavior, anatomy and physiological low-frequency signals could also contribute. Baseline removal deletes information indiscriminately, including any useful physiological level information. Conversely, residual performance after removing additive offsets could still arise from acquisition gain/noise or session-specific behavior. The observations do not separate these explanations because identity and session are confounded.

## 9. What remains unknown and the next experiment

The original ROC-AUC point estimates are reproducible descriptions of these recordings, with substantial user variation and conditional uncertainty. They do not establish persistent biometric identity, generalization across days, reliable security operating points, or artifact causation. New transformations were evaluated after observing the original benchmark, so independent confirmation is required rather than choosing a winning transformation from this test set.

The single most valuable experiment is a prospective, crossed multi-session collection: at least three days per participant, electrodes removed and reapplied between sessions, with placement/reference/contact quality and amplifier calibration recorded. Include a controlled reapplication within a day to distinguish reapplication from day effects; randomize placement/operator/order where practical and hold task/gaze instructions consistent. Enroll on session one, select preprocessing and thresholds using session two, lock the protocol, and evaluate once on session three with impostor identities absent from fitting. Prespecify a target FAR and acquire enough independent impostor attempts and sessions to estimate it. This tests whether identity separation persists while electrical acquisition conditions change.

## 10. Reproduction and evidence

Run `python run_baseline_robustness.py` after setting `EOG_RAW_DIR` as documented in README. Run `python -m unittest test_baseline_robustness.py` for data-free mathematical, causal and uncertainty tests. A nonempty output directory is rejected; use `--output results/baseline_robustness_replication` for a second preserved run. CSVs retain per-user metrics, baseline summaries, threshold deletions, matched assignments and comparisons. JSON retains both uncertainty methods, exact configuration, source checksums, and dependency versions. Previous authentication outputs remain unchanged; participant-level tables and all new figures stay in ignored local results.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(HERE / 'results'):
        parser.error('Participant-level outputs must stay under the ignored results/ directory.')
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is nonempty; choose a fresh --output to preserve results.')
    config = json.loads((HERE / 'robustness_config.json').read_text())
    auth_config = json.loads((HERE / 'auth_config.json').read_text())
    checksums = verify_data()
    matrices, y, windows, info = load_variants(config)
    split = lb.make_split(y, windows, 'chronological')
    lb.check_split(split, y, len(y))
    print('Raw data verified; running fixed transformation comparisons.', flush=True)
    records, details = known_experiments(matrices, y, split, config, auth_config['model'])
    # Check against existing outputs when available, without modifying them.
    previous_path = HERE / 'results/authentication/metrics.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())['known_impostor']['macro_per_user']
        for new, old in [('original', 'all_6'), ('means_only', 'means_only'), ('std_range_only', 'std_range_only')]:
            for metric in auth.METRIC_NAMES:
                assert np.isclose(auth.aggregate(records[new])[metric], previous[old][metric], atol=1e-12)
    print('Computing paired identity bootstrap and threshold deletions.', flush=True)
    stats, paired = uncertainty(records, details, y, config)
    deletions, threshold_rows = threshold_audit(details, y)
    print('Running matched seen/unseen impostor comparisons.', flush=True)
    matched_rows, assignments, matched = matched_protocol(matrices, y, split, config, auth_config['model'])
    baseline_rows = br.participant_summary(matrices['original'], y, windows, split)
    for row in baseline_rows:
        pid = row['participant']
        for name in ['original', 'means_only', 'std_range_only']:
            performance = next(r for r in records[name] if r['participant'] == pid)
            row[f'{name}_roc_auc'] = performance['roc_auc']
            row[f'{name}_eer'] = performance['eer']
    base_summary = {}
    for feature in ['A_mean', 'B_mean']:
        rr = [r for r in baseline_rows if r['feature'] == feature]
        base_summary[feature] = {
            'between_sd': rr[0]['between_sd'],
            'median_within_sd': float(np.median([r['within_sd'] for r in rr])),
            'median_within_between_ratio': float(np.median([r['within_between_ratio'] for r in rr])),
            'median_r2': float(np.median([r['r2'] for r in rr])),
            'positive_slopes': sum(r['slope_per_window'] > 0 for r in rr)}
    separation = [r['train_nearest_distance'] for r in baseline_rows if r['feature'] == 'A_mean']
    rho = spearmanr(separation, [r['roc_auc'] for r in records['means_only']]).statistic
    threshold_summary = {}
    for name in ['original', 'means_only', 'std_range_only']:
        rr = [r for r in threshold_rows if r['variant'] == name and r['deletion'] == 'delete_one_genuine']
        originals = {r['participant']: r for r in records[name]}
        threshold_summary[name] = {
            **{f'{stat}_genuine_deletion_{metric}_range': float(func([r[f'{metric}_max'] - r[f'{metric}_min'] for r in rr]))
               for stat, func in [('median', np.median), ('max', np.max)] for metric in ['far', 'frr']},
            'users_with_frr_change': sum(not np.isclose(r['frr_min'], originals[r['participant']]['frr']) or
                                         not np.isclose(r['frr_max'], originals[r['participant']]['frr']) for r in rr)}
    result = {'config': config, 'model_config': auth_config['model'],
              'raw_checksums': checksums,
              'environment': {'python': platform.python_version(), **{
                  package: importlib.metadata.version(package) for package in ['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'h5py']}},
              'source_hashes': {name: sha256(HERE / name) for name in [
                  'baseline.py', 'authentication.py', 'baseline_robustness.py', 'run_baseline_robustness.py',
                  'robustness_config.json', 'auth_config.json']},
              'dataset': {'participants': 27, 'windows': 918, 'windows_per_participant': 34},
              'baseline_summary': base_summary, 'uncertainty': stats, 'paired_comparisons': paired,
              'threshold_summary': threshold_summary, 'matched_impostors': matched,
              'separation_relationship': {'spearman_rho': float(rho), 'scope': 'exploratory, enrollment-only distances; shared neighbors'},
              'participant_spread': {f'{metric}_{name}': float(func([r[metric] for r in records['original']]))
                                     for metric in ['roc_auc', 'eer'] for name, func in [('min', min), ('max', max)]}}
    result['participant_spread']['auc_min'] = result['participant_spread'].pop('roc_auc_min')
    result['participant_spread']['auc_max'] = result['participant_spread'].pop('roc_auc_max')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_csv(out / 'per_user_metrics.csv', [{'variant': name, **r} for name, rr in records.items() for r in rr])
    write_csv(out / 'baseline_participants.csv', baseline_rows)
    write_csv(out / 'threshold_deletions.csv', deletions)
    write_csv(out / 'threshold_ranges.csv', threshold_rows)
    write_csv(out / 'matched_metrics.csv', matched_rows)
    write_csv(out / 'matched_identity_assignments.csv', assignments)
    plot_results(out, matrices, y, windows, baseline_rows, records, stats, threshold_rows, matched)
    report = make_report(result)
    (out / 'BASELINE_ROBUSTNESS_REPORT.md').write_text(report, encoding='utf-8')
    if out == DEFAULT_OUTPUT:
        # The repository-level report contains aggregate numbers only. Figure
        # targets remain local/ignored; sharing it does not publish those PNGs.
        root_report = report.replace('](baseline_', '](results/baseline_robustness/baseline_')
        (HERE / 'BASELINE_ROBUSTNESS_REPORT.md').write_text(root_report, encoding='utf-8')
    print(json.dumps({'transformation_auc': {name: m['roc_auc']['estimate'] for name, m in stats.items()},
                      'paired': paired, 'matched': matched['original'], 'baseline': base_summary,
                      'thresholds': threshold_summary['original']}, indent=2))
    print('Report and local evidence written to', out.relative_to(HERE))


if __name__ == '__main__':
    main()
