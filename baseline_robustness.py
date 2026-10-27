"""Signal transformations and conditional uncertainty for the EOG follow-up.

No data are loaded on import. Bootstrap units are complete identities, never
individual windows. Fitted scores remain fixed: intervals are conditional
sensitivity estimates and do not include retraining or session uncertainty.
"""

import numpy as np
from scipy.signal import lfilter

import authentication as auth
import baseline as lb


def transform_window(signal, method, cutoff_hz=.5, fs=250):
    """Apply a fixed, identity-blind transformation to one authentication trial.

    window_centered uses the completed two-second trial, not other windows.
    first_sample_reference is causal; the reference is available at sample 0.
    causal_dc_blocker resets each trial with output[0]=0 and uses only past
    samples: z[n]=x[n]-x[n-1]+alpha*z[n-1]. alpha is fixed, not learned.
    """
    x = np.asarray(signal, dtype=float)
    if method == "original":
        return x.copy()
    if method == "window_centered":
        return x - x.mean()
    if method == "first_sample_reference":
        return x - x[0]
    if method == "causal_dc_blocker":
        alpha = np.exp(-2 * np.pi * cutoff_hz / fs)
        # Difference first so an arbitrary initial electrical level cannot
        # create a large filter startup transient. Every trial resets state.
        return lfilter([1.0], [1.0, -alpha], np.diff(x, prepend=x[0]))
    raise ValueError(f"Unknown transformation: {method}")


def transformed_features(a, b, method, cutoff_hz=.5):
    values = np.asarray(lb.window_features(
        transform_window(a, method, cutoff_hz),
        transform_window(b, method, cutoff_hz)))
    if method == "window_centered":
        # Mathematical means are exactly zero. Do not let StandardScaler
        # amplify floating-point cancellation residuals into fake predictors.
        values[[0, 3]] = 0.0
    return values


def participant_summary(X, y, windows, split):
    """Descriptive raw-unit baseline/drift statistics, no model preprocessing."""
    pids = np.unique(y)
    centers = np.array([X[y == p][:, [0, 3]].mean(axis=0) for p in pids])
    # Between-recording SD sets a descriptive common scale, never a model input.
    between = centers.std(axis=0, ddof=1)
    train_centers = np.array([X[split['train']][y[split['train']] == p][:, [0, 3]].mean(axis=0)
                              for p in pids])
    train_scale = train_centers.std(axis=0, ddof=1)
    standardized = train_centers / train_scale
    distances = np.linalg.norm(standardized[:, None] - standardized[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    rows = []
    for i, pid in enumerate(pids):
        ordered = np.where(y == pid)[0]
        ordered = ordered[np.argsort(windows[ordered])]
        for j, feature_col in enumerate([0, 3]):
            values = X[ordered, feature_col]
            trend = np.polyfit(windows[ordered], values, 1)
            fitted = np.polyval(trend, windows[ordered])
            variance = np.sum((values - values.mean()) ** 2)
            rows.append({
                "participant": int(pid), "feature": lb.FEATURE_NAMES[feature_col],
                "mean": float(values.mean()), "within_sd": float(values.std(ddof=1)),
                "between_sd": float(between[j]),
                "within_between_ratio": float(values.std(ddof=1) / between[j]),
                "slope_per_window": float(trend[0]),
                "r2": float(1 - np.sum((values - fitted) ** 2) / variance) if variance else 0.,
                "late_minus_early": float(np.median(values[-7:]) - np.median(values[:7])),
                "train_nearest_distance": float(distances[i].min()),
                "train_nearest_participant": int(pids[np.argmin(distances[i])]),
            })
    return rows


def bootstrap_counts(n_participants, draws, seed):
    """Shared multinomial identity counts for paired resampling."""
    return np.random.default_rng(seed).multinomial(
        n_participants, np.full(n_participants, 1 / n_participants), size=draws)


def interval(values):
    values = np.asarray(values, dtype=float)
    return {"low": float(np.quantile(values, .025)),
            "high": float(np.quantile(values, .975))}


def cluster_bootstrap(records, counts, metric):
    """Resample authorized-user tasks with complete test sequences intact."""
    ordered = sorted(records, key=lambda r: r['participant'])
    values = np.array([r[metric] for r in ordered])
    return counts @ values / counts.sum(axis=1)


def identity_weighted_metrics(detail, y, target, pids, counts):
    """Reweight identities in both genuine-task and impostor roles.

    Returns per-draw metrics for one frozen verifier. Tied score groups are
    accumulated together, so weighted ROC EER agrees with empirical ROC EER.
    Each identity's complete seven-window sequence receives one shared weight.
    """
    scores = detail['test_scores']
    truth = detail['y_test'].astype(bool)
    sources = y[detail['test_indices']]
    lookup = {p: i for i, p in enumerate(pids)}
    negative_ids = np.array([lookup[p] for p in sources[~truth]])
    weights = counts[:, negative_ids].astype(float)
    neg_sum = weights.sum(axis=1)
    # Degenerate single-identity bootstrap draws cannot define impostor scores.
    valid = neg_sum > 0
    safe_sum = np.where(valid, neg_sum, 1.)
    positive = scores[truth]
    negative = scores[~truth]
    comparisons = ((positive[:, None] > negative[None, :]).astype(float) +
                   .5 * (positive[:, None] == negative[None, :])).mean(axis=0)
    auc = weights @ comparisons / safe_sum
    accepted = scores >= detail['threshold']
    far = weights @ accepted[~truth].astype(float) / safe_sum
    frr = np.full(len(counts), 1 - accepted[truth].mean())

    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.where(np.diff(scores[order]) != 0)[0], len(scores) - 1]
    sorted_truth = truth[order]
    per_observation = np.zeros((len(counts), len(scores)))
    for k, row in enumerate(order):
        if not truth[row]:
            per_observation[:, k] = counts[:, lookup[sources[row]]]
    fpr = np.c_[np.zeros(len(counts)), np.cumsum(per_observation, axis=1)[:, ends] / safe_sum[:, None]]
    fnr = np.r_[1., 1 - np.cumsum(sorted_truth)[ends] / truth.sum()]
    diff = fpr - fnr[None, :]
    hi = np.argmax(diff >= 0, axis=1)
    lo = np.maximum(hi - 1, 0)
    index = np.arange(len(counts))
    lower, upper = diff[index, lo], diff[index, hi]
    denom = upper - lower
    mix = np.divide(-lower, denom, out=np.zeros_like(lower), where=denom != 0)
    eer = fpr[index, lo] + mix * (fpr[index, hi] - fpr[index, lo])
    return {"roc_auc": np.where(valid, auc, np.nan),
            "eer": np.where(valid, eer, np.nan),
            "far": np.where(valid, far, np.nan), "frr": frr}


def crossed_identity_bootstrap(details, y, pids, counts):
    """Shared participant resampling in both task and impostor roles."""
    totals = {k: np.zeros(len(counts)) for k in ('roc_auc', 'eer', 'far', 'frr')}
    denominator = np.zeros(len(counts))
    for i, target in enumerate(pids):
        metrics = identity_weighted_metrics(details[int(target)], y, target, pids, counts)
        valid = np.isfinite(metrics['roc_auc'])
        w = counts[:, i] * valid
        denominator += w
        for name, values in metrics.items():
            totals[name] += np.nan_to_num(values) * w
    return {name: np.divide(value, denominator, out=np.full_like(value, np.nan),
                            where=denominator > 0) for name, value in totals.items()}


def threshold_deletions(detail, y):
    """Delete one validation genuine or one complete validation impostor ID.

    Frozen model scores are reused. Test results describe sensitivity only;
    they never choose a threshold or replace the original benchmark threshold.
    """
    truth = detail['y_validation']
    scores = detail['validation_scores']
    sources = y[detail['validation_indices']]
    scenarios = []
    for index in np.where(truth == 1)[0]:
        keep = np.ones(len(truth), dtype=bool)
        keep[index] = False
        scenarios.append(('delete_one_genuine', int(index), keep))
    for pid in np.unique(sources[truth == 0]):
        scenarios.append(('delete_one_impostor_identity', int(pid), sources != pid))
    rows = []
    for kind, deleted, keep in scenarios:
        threshold, _ = auth.threshold_from_validation(truth[keep], scores[keep])
        test = detail['test_scores'] >= threshold
        rows.append({"deletion": kind, "deleted_validation_unit": deleted,
                     "threshold": float(threshold),
                     "far": float(test[detail['y_test'] == 0].mean()),
                     "frr": float(1 - test[detail['y_test'] == 1].mean())})
    return rows


def matched_impostor_indices(y, split, target, groups, seen):
    """Equal training sizes and identical val/test panels in paired conditions.

    Unseen uses the original 16/5/5 identity groups, restricted to chronological
    20/7/7 windows. Seen replaces five of the 16 training identities with the
    five test identities. Validation identities remain unseen in both arms.
    """
    training = groups['train']
    if seen:
        training = np.r_[training[:11], groups['test']]
    selected = {'train': training, 'val': groups['val'], 'test': groups['test']}
    return {name: np.asarray(split[name])[np.isin(y[split[name]], np.r_[target, ids])]
            for name, ids in selected.items()}
