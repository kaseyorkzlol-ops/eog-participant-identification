"""Core methods for within-session EOG one-to-rest verification.

This module contains no file I/O and does not load the dataset on import.  The
runner in ``run_authentication.py`` supplies the six-feature matrix produced by
``baseline.build_dataset()``.  Keeping the core separate makes threshold and
split behavior testable with synthetic data when the private recordings are
not available.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler


METRIC_NAMES = ("roc_auc", "eer", "far", "frr", "balanced_accuracy",
                "precision", "recall", "pr_auc")


def equal_error_rate(y_true, scores):
    """Return linearly interpolated EER and a finite empirical threshold.

    The interpolation is performed where empirical FPR and FNR cross.  This is
    a descriptive continuous-score metric.  When called on validation scores,
    the returned threshold may be frozen as the operating threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("EER requires both genuine (1) and impostor (0) observations")
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    exact = np.where(np.isclose(diff, 0.0, atol=1e-15))[0]
    if len(exact):
        i = int(exact[0])
        return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])
    upper = np.where(diff > 0)[0]
    if not len(upper):
        i = int(np.argmin(np.abs(diff)))
        return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])
    hi = int(upper[0])
    lo = hi - 1
    weight = -diff[lo] / (diff[hi] - diff[lo])
    eer = fpr[lo] + weight * (fpr[hi] - fpr[lo])
    # ROC curves begin with an infinite sentinel threshold. Interpolating the
    # error *rate* is well defined, but interpolating score thresholds is not.
    # Freeze the finite empirical threshold closest to the crossing instead.
    nearest = lo if abs(diff[lo]) <= abs(diff[hi]) else hi
    if not np.isfinite(thresholds[nearest]):
        nearest = hi
    return float(eer), float(thresholds[nearest])


def threshold_from_validation(y_validation, validation_scores):
    """Select the operating threshold from validation data only."""
    eer, threshold = equal_error_rate(y_validation, validation_scores)
    return threshold, eer


def binary_metrics(y_true, scores, threshold):
    """Compute verification metrics at a frozen threshold plus score metrics."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if fp + tn else math.nan
    frr = fn / (fn + tp) if fn + tp else math.nan
    eer, eer_threshold = equal_error_rate(y_true, scores)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far": float(far),
        "frr": float(frr),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_genuine": int((y_true == 1).sum()),
        "n_impostor": int((y_true == 0).sum()),
    }, predicted


def fit_verifier(X_train, y_train, seed, model_config):
    """Fit train-only scaling and class-balanced logistic regression."""
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        C=float(model_config.get("C", 1.0)),
        class_weight=model_config.get("class_weight", "balanced"),
        max_iter=int(model_config.get("max_iter", 10000)),
        random_state=int(seed),
    )
    model.fit(scaler.transform(X_train), y_train)
    return scaler, model


def evaluate_target(X, y, target, indices, seed, model_config):
    """Fit one target verifier, select on validation, evaluate once on test."""
    train_idx, val_idx, test_idx = (np.asarray(indices[k], dtype=int)
                                    for k in ("train", "val", "test"))
    y_train = (y[train_idx] == target).astype(int)
    y_val = (y[val_idx] == target).astype(int)
    y_test = (y[test_idx] == target).astype(int)
    for name, binary in (("train", y_train), ("validation", y_val), ("test", y_test)):
        if set(np.unique(binary)) != {0, 1}:
            raise ValueError(f"target {target}: {name} partition lacks a class")

    scaler, model = fit_verifier(X[train_idx], y_train, seed, model_config)
    val_scores = model.predict_proba(scaler.transform(X[val_idx]))[:, 1]
    threshold, validation_eer = threshold_from_validation(y_val, val_scores)
    test_scores = model.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    metrics, predicted = binary_metrics(y_test, test_scores, threshold)
    record = {
        "participant": int(target),
        "threshold": float(threshold),
        "validation_eer": float(validation_eer),
        **metrics,
    }
    detail = {
        "test_indices": test_idx, "y_test": y_test, "test_scores": test_scores,
        "test_predicted": predicted, "validation_indices": val_idx,
        "y_validation": y_val, "validation_scores": val_scores,
        "scaler_mean": scaler.mean_.copy(), "scaler_scale": scaler.scale_.copy(),
    }
    return record, detail


def run_known_impostor(X, y, split, participant_ids, seed, model_config):
    """One-vs-rest protocol where every impostor identity is known in training."""
    records, details = [], {}
    for target in participant_ids:
        record, detail = evaluate_target(X, y, target, split, seed, model_config)
        records.append(record)
        details[int(target)] = detail
    return records, details


def make_unseen_identity_split(target, participant_ids, seed, counts=(16, 5, 5)):
    """Partition impostor identities deterministically with no identity overlap."""
    impostors = np.array([p for p in participant_ids if p != target], dtype=int)
    if sum(counts) != len(impostors):
        raise ValueError("unseen-impostor identity counts must use every non-target identity")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(target)]))
    impostors = rng.permutation(impostors)
    a, b = counts[0], counts[0] + counts[1]
    return {"train": impostors[:a], "val": impostors[a:b], "test": impostors[b:]}


def unseen_observation_indices(y, chronological_split, target, identity_groups):
    """Combine chronological genuine rows with all rows of disjoint impostor IDs."""
    out = {}
    for name in ("train", "val", "test"):
        genuine = np.asarray(chronological_split[name], dtype=int)
        genuine = genuine[y[genuine] == target]
        impostor = np.where(np.isin(y, identity_groups[name]))[0]
        out[name] = np.concatenate([genuine, impostor])
    return out


def run_unseen_impostor(X, y, chronological_split, participant_ids, seed,
                        repeats, identity_counts, model_config):
    """Evaluate disjoint impostor-identity groups over deterministic repeats."""
    records, details, assignments = [], {}, []
    for repeat in range(int(repeats)):
        repeat_seed = int(seed) + repeat
        for target in participant_ids:
            groups = make_unseen_identity_split(target, participant_ids, repeat_seed,
                                                tuple(identity_counts))
            indices = unseen_observation_indices(y, chronological_split, target, groups)
            record, detail = evaluate_target(X, y, target, indices, repeat_seed, model_config)
            record["repeat"] = repeat
            record["partition_seed"] = repeat_seed
            records.append(record)
            details[(repeat, int(target))] = detail
            for split_name, ids in groups.items():
                for impostor_id in ids:
                    assignments.append({"repeat": repeat, "participant": int(target),
                                        "split": split_name, "impostor_participant": int(impostor_id)})
    return records, details, assignments


def aggregate(records):
    """Macro-average per-verifier metrics; each authorized user has equal weight."""
    out = {"n_verifiers": len(records)}
    for name in METRIC_NAMES:
        values = np.array([r[name] for r in records], dtype=float)
        out[name] = float(values.mean())
        out[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    out["total_genuine_observations"] = int(sum(r["n_genuine"] for r in records))
    out["total_impostor_observations"] = int(sum(r["n_impostor"] for r in records))
    return out


def pooled_score_metrics(details):
    """Pool continuous scores; thresholds remain user-specific for predictions."""
    values = list(details.values())
    y_true = np.concatenate([d["y_test"] for d in values])
    scores = np.concatenate([d["test_scores"] for d in values])
    predicted = np.concatenate([d["test_predicted"] for d in values])
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "far": float(fp / (fp + tn)), "frr": float(fn / (fn + tp)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "note": "Continuous scores pooled across verifiers; thresholded predictions use each verifier's validation-selected threshold.",
    }
