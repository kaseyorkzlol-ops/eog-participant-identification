"""Run the within-session EOG verification benchmark.

Outputs are written under ``results/authentication/`` and are intentionally
git-ignored because they contain participant-level results. This script does
not alter the existing multiclass experiments or their outputs.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_curve

import authentication as auth
import baseline as lb


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "auth_config.json")
OUT_DIR = os.path.join(lb.RESULTS_ROOT, "authentication")
LABELS = {"all_6": "All six features", "means_only": "Means only",
          "std_range_only": "Std/range only (both means excluded)"}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_dict_csv(path, records, fields=None):
    if not records:
        return
    if fields is None:
        fields = list(records[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def summarize_chronological_drift(X, y, window_index):
    """Median per-user linear-trend R-squared for each feature."""
    summary = {}
    for column, feature in enumerate(lb.FEATURE_NAMES):
        values = []
        for participant in lb.PARTICIPANT_IDS:
            rows = np.where(y == participant)[0]
            order = rows[np.argsort(window_index[rows])]
            t = window_index[order].astype(float)
            v = X[order, column]
            correlation = np.corrcoef(t, v)[0, 1]
            values.append(float(correlation ** 2) if np.isfinite(correlation) else 0.0)
        summary[feature] = {"median_r2": float(np.median(values)),
                            "n_participants_r2_above_0_5": int(np.sum(np.asarray(values) > .5))}
    return summary


def save_roc_plot(records, details, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(records, key=lambda r: (r["roc_auc"], r["participant"]))
    chosen = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    chosen_ids = {r["participant"] for r in chosen}
    grid = np.linspace(0, 1, 201)
    curves = []
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for record in records:
        detail = details[record["participant"]]
        fpr, tpr, _ = roc_curve(detail["y_test"], detail["test_scores"])
        curves.append(np.interp(grid, fpr, tpr))
        if record["participant"] not in chosen_ids:
            ax.plot(fpr, tpr, color="#aeb7c2", alpha=.32, lw=.8)
    colors = ["#b23a48", "#2a78d6", "#25855a"]
    names = ["lowest", "median", "highest"]
    for record, color, name in zip(chosen, colors, names):
        detail = details[record["participant"]]
        fpr, tpr, _ = roc_curve(detail["y_test"], detail["test_scores"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} AUC: P{record['participant']:02d} ({record['roc_auc']:.3f})")
    ax.plot(grid, np.mean(curves, axis=0), color="#111111", lw=2.4,
            label="mean interpolated ROC")
    ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1)
    ax.set(xlabel="False acceptance rate", ylabel="True acceptance rate",
           title="Known-impostor test ROC curves, all six features")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_eer_distribution(all_records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for position, name in enumerate(LABELS):
        values = np.array([r["eer"] for r in all_records[name]])
        ax.boxplot(values, positions=[position], widths=.45, showfliers=False,
                   medianprops={"color": "#222222"})
        ax.scatter(position + rng.uniform(-.14, .14, len(values)), values,
                   s=20, alpha=.65, color="#2a78d6", linewidths=0)
    ax.set_xticks(range(len(LABELS)), [LABELS[k] for k in LABELS], rotation=12)
    ax.set_ylabel("Test equal error rate")
    ax.set_title("Per-participant EER: chronological known-impostor verification")
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_ablation_plot(aggregates, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("roc_auc", "ROC-AUC", (0, 1)), ("eer", "EER", (0, .6)),
               ("far", "FAR", (0, .6)), ("frr", "FRR", (0, .6))]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    colors = ["#2a78d6", "#eb8f34", "#25855a"]
    for ax, (metric, title, ylim) in zip(axes.ravel(), metrics):
        values = [aggregates[name][metric] for name in LABELS]
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(values)), [LABELS[k] for k in LABELS], rotation=15, fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        for i, value in enumerate(values):
            ax.text(i, value + (ylim[1] - ylim[0]) * .02, f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Known-impostor verification: macro average across authorized users")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_far_frr_plot(records, details, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = sorted(records, key=lambda r: (r["roc_auc"], r["participant"]))[len(records) // 2]
    detail = details[selected["participant"]]
    fpr, tpr, thresholds = roc_curve(detail["y_validation"], detail["validation_scores"],
                                     drop_intermediate=False)
    finite = np.isfinite(thresholds)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(thresholds[finite], fpr[finite], label="Validation FAR", color="#b23a48", lw=2)
    ax.plot(thresholds[finite], (1 - tpr)[finite], label="Validation FRR", color="#2a78d6", lw=2)
    ax.axvline(selected["threshold"], color="#222222", ls="--", lw=1.5,
               label=f"selected threshold {selected['threshold']:.3f}")
    ax.set(xlabel="Acceptance threshold (genuine probability)", ylabel="Error rate",
           title=f"Validation operating point: median-AUC P{selected['participant']:02d}")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_user_plot(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = sorted(records, key=lambda r: r["participant"])
    x = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, [r["roc_auc"] for r in records], marker="o", ms=4, lw=1.4,
            label="ROC-AUC", color="#2a78d6")
    ax.plot(x, [r["eer"] for r in records], marker="s", ms=4, lw=1.4,
            label="EER", color="#b23a48")
    ax.set_xticks(x, [f"P{r['participant']:02d}" for r in records], rotation=60, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test metric")
    ax.set_title("Per-user known-impostor verification, all six features")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_report(info, known_agg, random_agg, unseen_agg, drift, records_by_feature,
                repeats, path):
    all6 = known_agg["all_6"]
    means = known_agg["means_only"]
    spread = known_agg["std_range_only"]
    unseen = unseen_agg["all_6"]
    best = max(records_by_feature["all_6"], key=lambda r: (r["roc_auc"], -r["participant"]))
    worst = min(records_by_feature["all_6"], key=lambda r: (r["roc_auc"], r["participant"]))
    rows = [("All six features", all6), ("Means only", means),
            ("Std/range only", spread),
            (f"Unseen impostors (all six; {repeats} repeats)", unseen)]
    table = "\n".join(
        f"| {label} | {m['roc_auc']:.3f} | {m['eer']:.3f} | {m['far']:.3f} | {m['frr']:.3f} |"
        for label, m in rows)
    text = f"""# EOG Authentication Benchmark Report

Generated from the local raw recordings by `python run_authentication.py`.

## Dataset audit

- {len(info)} participants; {sum(r['n_windows'] for r in info)} non-overlapping windows.
- Every participant has {info[0]['n_windows']} chronological 2-second windows; the primary split is 20 train, 7 validation, 7 test.
- Inputs are raw two-channel EOG recordings. This benchmark extracts six window summaries.
- One recording session exists per participant. Windows are correlated observations, not independent sessions.

## Primary results

Values are macro averages across authorized-user verifiers. EER is calculated from continuous test scores. FAR and FRR use the threshold selected on validation scores only.

| Experiment | ROC-AUC | EER | FAR | FRR |
|---|---:|---:|---:|---:|
{table}

Always rejecting in the primary known-impostor test gives conventional accuracy 182/189 = 96.3%, so raw accuracy is not used as the principal result.

## Method

For each authorized participant, a StandardScaler and logistic regression (`C=1`, balanced class weights) are fitted only on training observations. EER is interpolated between ROC points; the operating threshold is the nearest finite empirical validation threshold to that crossing and is frozen before test scoring. Test EER is descriptive and does not set the operating threshold. Macro results weight each authorized user equally; pooled results are saved separately.

The unseen-impostor protocol assigns the other 26 identities to disjoint groups of 16 train, 5 validation, and 5 test identities. All 34 windows of an assigned impostor identity remain in that identity's one group. The genuine user's windows still use the chronological 20/7/7 split. {repeats} deterministic identity partitions quantify partition variability, but reuse of the same genuine recording means these are not independent replications.

## Observations

- Known-impostor all-six macro ROC-AUC is {all6['roc_auc']:.3f}, with EER {all6['eer']:.3f}.
- Means-only macro ROC-AUC is {means['roc_auc']:.3f}; excluding both means gives {spread['roc_auc']:.3f}. Removing feature means is an ablation, not correction of raw-electrode DC offset.
- Across users, all-six ROC-AUC ranges from {worst['roc_auc']:.3f} (P{worst['participant']:02d}) to {best['roc_auc']:.3f} (P{best['participant']:02d}). These differences are based on seven genuine test windows each.
- Unseen-impostor all-six macro ROC-AUC is {unseen['roc_auc']:.3f}. This tests attacker-identity generalization, not cross-session reliability.
- A secondary random-window split gives all-six ROC-AUC {random_agg['all_6']['roc_auc']:.3f}, versus {all6['roc_auc']:.3f} under chronological holdout. Random windows are temporally interleaved and are an optimism diagnostic, not an independent replication.
- Median per-participant linear-trend R-squared is {drift['A_mean']['median_r2']:.2f} for A_mean and {drift['B_mean']['median_r2']:.2f} for B_mean, versus {drift['A_std']['median_r2']:.2f}-{drift['B_range']['median_r2']:.2f} for the four spread features. The strong mean drift reinforces the recording-artifact concern.

## Scientific limits

These results can establish only within-session separation under this protocol. They cannot show that the signal is a persistent biometric, distinguish physiological identity from electrode placement/amplifier/session artifacts, estimate performance on a later day, or support deployment as a security system. Adjacent windows are correlated, and the small genuine test set makes per-user error rates coarse.

## Most valuable follow-up experiment

Collect at least three independently removed-and-reapplied-electrode sessions per participant on different days. Enroll on one session, select thresholds on a second, and test once on a third, with impostors absent from model fitting. That design directly tests persistence and separates identity from session-specific acquisition artifacts.
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    os.makedirs(OUT_DIR, exist_ok=True)
    X_all, y, window_index, _, info = lb.build_dataset()
    diagnostics = lb.check_dataset(X_all, y)
    split = lb.make_split(y, window_index, "chronological")
    lb.check_split(split, y, len(y))
    assert all(lb.split_sizes(r["n_windows"]) == (20, 7, 7) for r in info)

    records_by_feature, details_by_feature = {}, {}
    known_aggregates, known_pooled = {}, {}
    per_user_rows, prediction_rows = [], []
    for feature_set in config["feature_sets"]:
        X = lb.select_features(X_all, lb.FEATURE_SETS[feature_set])
        records, details = auth.run_known_impostor(
            X, y, split, lb.PARTICIPANT_IDS, config["seed"], config["model"])
        records_by_feature[feature_set], details_by_feature[feature_set] = records, details
        known_aggregates[feature_set] = auth.aggregate(records)
        known_pooled[feature_set] = auth.pooled_score_metrics(details)
        for record in records:
            per_user_rows.append({"protocol": "known_impostor", "feature_set": feature_set, **record})
            detail = details[record["participant"]]
            for idx, truth, score, pred in zip(detail["test_indices"], detail["y_test"],
                                                detail["test_scores"], detail["test_predicted"]):
                prediction_rows.append({"feature_set": feature_set,
                                        "authorized_participant": record["participant"],
                                        "row": int(idx), "source_participant": int(y[idx]),
                                        "window_index": int(window_index[idx]),
                                        "is_genuine": int(truth), "score": float(score),
                                        "accepted": int(pred), "threshold": record["threshold"]})

    # Secondary temporal-optimism diagnostic. This never replaces the primary
    # chronological protocol and uses one fixed, documented split seed.
    random_split = lb.make_split(y, window_index, "random", seed=config["seed"])
    lb.check_split(random_split, y, len(y))
    random_aggregates = {}
    for feature_set in config["feature_sets"]:
        X = lb.select_features(X_all, lb.FEATURE_SETS[feature_set])
        records, _ = auth.run_known_impostor(
            X, y, random_split, lb.PARTICIPANT_IDS, config["seed"], config["model"])
        random_aggregates[feature_set] = auth.aggregate(records)
        for record in records:
            per_user_rows.append({"protocol": "random_window_known_impostor",
                                  "feature_set": feature_set, **record})

    unseen_aggregates, unseen_pooled, assignments = {}, {}, []
    unseen_cfg = config["unseen_impostor"]
    counts = (unseen_cfg["train_identities"], unseen_cfg["validation_identities"],
              unseen_cfg["test_identities"])
    for feature_set in config["feature_sets"]:
        X = lb.select_features(X_all, lb.FEATURE_SETS[feature_set])
        records, details, feature_assignments = auth.run_unseen_impostor(
            X, y, split, lb.PARTICIPANT_IDS, config["seed"], unseen_cfg["repeats"],
            counts, config["model"])
        unseen_aggregates[feature_set] = auth.aggregate(records)
        unseen_pooled[feature_set] = auth.pooled_score_metrics(details)
        for record in records:
            per_user_rows.append({"protocol": "unseen_impostor", "feature_set": feature_set, **record})
        if feature_set == "all_6":
            assignments = feature_assignments

    fields = (["protocol", "feature_set", "repeat", "partition_seed", "participant",
               "threshold", "validation_eer"] + list(auth.METRIC_NAMES) +
              ["eer_threshold", "tn", "fp", "fn", "tp", "n_genuine", "n_impostor"])
    write_dict_csv(os.path.join(OUT_DIR, "per_user_metrics.csv"), per_user_rows, fields)
    write_dict_csv(os.path.join(OUT_DIR, "known_impostor_test_predictions.csv"), prediction_rows)
    write_dict_csv(os.path.join(OUT_DIR, "unseen_impostor_identity_assignments.csv"), assignments)
    aggregate_rows = []
    for protocol, values in (("known_impostor", known_aggregates),
                             ("random_window_known_impostor", random_aggregates),
                             ("unseen_impostor", unseen_aggregates)):
        for feature_set, metrics in values.items():
            aggregate_rows.append({"protocol": protocol, "feature_set": feature_set, **metrics})
    write_dict_csv(os.path.join(OUT_DIR, "aggregate_metrics.csv"), aggregate_rows)

    drift = summarize_chronological_drift(X_all, y, window_index)
    output = {
        "scope": "within-session verification; not evidence of a persistent biometric",
        "config": config,
        "dataset": {"n_participants": len(info), "n_windows": len(y),
                    "features": lb.FEATURE_NAMES, "diagnostics": diagnostics,
                    "split": {"genuine_train": 20, "genuine_validation": 7, "genuine_test": 7}},
        "threshold_selection": "Nearest finite empirical validation threshold to the interpolated EER crossing; frozen for test FAR/FRR.",
        "known_impostor": {"macro_per_user": known_aggregates, "pooled": known_pooled,
                           "always_reject_accuracy": 26 / 27},
        "random_window_known_impostor": {
            "macro_per_user": random_aggregates,
            "note": "Secondary temporal-optimism diagnostic using one fixed random split; not an independent replication."},
        "chronological_drift": drift,
        "unseen_impostor": {"macro_per_user_repeat": unseen_aggregates,
                            "pooled": unseen_pooled,
                            "note": "Impostor identities are disjoint; genuine recordings are reused across repeats."},
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(json_ready(output), handle, indent=2)
    with open(os.path.join(OUT_DIR, "config_snapshot.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    save_roc_plot(records_by_feature["all_6"], details_by_feature["all_6"],
                  os.path.join(OUT_DIR, "roc_curves_known_all6.png"))
    save_eer_distribution(records_by_feature, os.path.join(OUT_DIR, "eer_distribution.png"))
    save_ablation_plot(known_aggregates, os.path.join(OUT_DIR, "feature_ablation_comparison.png"))
    save_far_frr_plot(records_by_feature["all_6"], details_by_feature["all_6"],
                      os.path.join(OUT_DIR, "far_frr_validation_tradeoff.png"))
    save_per_user_plot(records_by_feature["all_6"], os.path.join(OUT_DIR, "per_user_auc_eer.png"))
    lb.save_drift_png(X_all, y, window_index,
                      os.path.join(OUT_DIR, "chronological_feature_drift.png"))
    make_report(info, known_aggregates, random_aggregates, unseen_aggregates, drift,
                records_by_feature, unseen_cfg["repeats"],
                os.path.join(OUT_DIR, "REPORT.md"))

    print("WITHIN-SESSION AUTHENTICATION BENCHMARK")
    print(f"{'Experiment':<43}{'ROC-AUC':>9}{'EER':>9}{'FAR':>9}{'FRR':>9}")
    for name in LABELS:
        m = known_aggregates[name]
        print(f"{LABELS[name]:<43}{m['roc_auc']:>9.3f}{m['eer']:>9.3f}{m['far']:>9.3f}{m['frr']:>9.3f}")
    m = unseen_aggregates["all_6"]
    unseen_label = f"Unseen impostors (all six; {unseen_cfg['repeats']} repeats)"
    print(f"{unseen_label:<43}{m['roc_auc']:>9.3f}{m['eer']:>9.3f}{m['far']:>9.3f}{m['frr']:>9.3f}")
    print("\nThese are within-session results, not evidence of cross-session reliability.")
    print("Artifacts written to", os.path.relpath(OUT_DIR, HERE))


if __name__ == "__main__":
    main()
