"""
Inspect the coefficients of the six-feature chronological logistic-regression baseline.

The features are standardized with the training-set scaler, so coefficients are
in "per training-SD" units and can be compared across features. Coefficient
magnitude describes how strongly this fitted linear classifier uses a
standardized feature; it is NOT a causal or physiological importance measure,
and correlated features can share or trade weight.

Note on multinomial logistic regression: adding the same constant to one
feature's coefficient in every class leaves all predictions unchanged. With
the L2 penalty, scikit-learn's solution has each feature's coefficients summing
to ~0 across the 27 classes, so a class's coefficient is relative to the
average class, not an absolute effect.

Run:  python run_coefficients.py      -> results/coefficients/
"""

import json
import os

import numpy as np

import baseline as lb

OUT_DIR = os.path.join(lb.RESULTS_ROOT, "coefficients")
FOCUS = [12, 13, 25]   # classes selected for focused inspection (chosen after the baseline analysis)


def fit_chronological_baseline():
    """Return (scaler, model, X, y, split) for the six-feature chronological baseline."""
    X, y, window_index, _, _ = lb.build_dataset()
    split = lb.make_split(y, window_index, "chronological")
    lb.check_split(split, y, len(y))
    scaler, model = lb.train_model(X[split["train"]], y[split["train"]])
    return scaler, model, X, y, split


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    scaler, model, X, y, split = fit_chronological_baseline()
    F = lb.FEATURE_NAMES

    # Row i of coef_ belongs to class model.classes_[i]; check that this is P01..P27 in order.
    classes = model.classes_.tolist()
    assert classes == lb.PARTICIPANT_IDS, classes
    coef = model.coef_                      # shape (27, 6)
    assert coef.shape == (27, len(F))

    # the rows reproduce the model's own predictions when used by hand
    Z = scaler.transform(X[split["test"]])
    manual = np.array(classes)[np.argmax(Z @ coef.T + model.intercept_, axis=1)]
    assert np.array_equal(manual, model.predict(Z))

    lb.write_csv(os.path.join(OUT_DIR, "lr_coefficients_all6_chronological.csv"),
                 ["participant", "intercept"] + F,
                 [[pid, round(float(model.intercept_[i]), 6)] + [round(float(c), 6) for c in coef[i]]
                  for i, pid in enumerate(classes)])

    # aggregate per feature
    abs_c = np.abs(coef)
    order = np.argsort(-abs_c.mean(axis=0))
    feat_rows = []
    for rank, j in enumerate(order, 1):
        feat_rows.append([F[j], rank, round(float(abs_c[:, j].mean()), 4), round(float(np.median(abs_c[:, j])), 4),
                          round(float(abs_c[:, j].max()), 4), classes[int(np.argmax(abs_c[:, j]))],
                          round(float(np.sqrt((coef[:, j] ** 2).sum())), 4), round(float(coef[:, j].sum()), 6)])
    lb.write_csv(os.path.join(OUT_DIR, "lr_coefficient_magnitude_by_feature.csv"),
                 ["feature", "rank_by_mean_abs", "mean_abs_coef", "median_abs_coef", "max_abs_coef",
                  "participant_with_max_abs", "l2_norm_over_classes", "sum_over_classes"], feat_rows)

    # share of each class's total |coef| that goes to the two mean features
    mean_cols = [F.index("A_mean"), F.index("B_mean")]
    mean_share = abs_c[:, mean_cols].sum(axis=1) / abs_c.sum(axis=1)

    # focus classes: coefficients next to the class's standardized TRAINING feature means
    Ztr = scaler.transform(X[split["train"]])
    ytr = y[split["train"]]
    focus = {}
    for pid in FOCUS:
        i = classes.index(pid)
        focus[f"P{pid:02d}"] = {
            "coefficients": {F[j]: round(float(coef[i, j]), 3) for j in range(len(F))},
            "standardized_training_mean": {F[j]: round(float(Ztr[ytr == pid, j].mean()), 3)
                                           for j in range(len(F))},
            "share_of_abs_coef_on_mean_features": round(float(mean_share[i]), 3),
        }

    summary = {
        "model": "six-feature chronological baseline (baseline.train_model)",
        "class_order_verified": True,
        "feature_ranking_by_mean_abs_coef": [r[0] for r in feat_rows],
        "mean_abs_coef": {r[0]: r[2] for r in feat_rows},
        "share_of_total_abs_coef_on_mean_features": round(float(abs_c[:, mean_cols].sum() / abs_c.sum()), 4),
        "per_class_share_on_mean_features": {"median": round(float(np.median(mean_share)), 3),
                                             "min": round(float(mean_share.min()), 3),
                                             "max": round(float(mean_share.max()), 3)},
        "focus_classes": focus,
        "max_abs_column_sum": float(np.abs(coef.sum(axis=0)).max()),
    }
    with open(os.path.join(OUT_DIR, "lr_coefficient_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- heatmap ----
    lim = np.ceil(abs_c.max())
    fig, ax = plt.subplots(figsize=(6.5, 9))
    im = ax.imshow(coef, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(F)), F, rotation=30, fontsize=9)
    ax.set_yticks(range(27), [f"P{p:02d}" + ("  *" if p in FOCUS else "") for p in classes], fontsize=8)
    for i in range(27):
        for j in range(len(F)):
            ax.text(j, i, f"{coef[i, j]:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(coef[i, j]) > lim * 0.55 else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="coefficient (standardized feature)")
    ax.set_title("Logistic regression coefficients, six-feature chronological baseline\n"
                 "rows = participant class, * = class selected for focused inspection", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lr_coefficient_heatmap.png"), dpi=150)
    plt.close(fig)

    print("Feature ranking by mean |coef| across 27 classes:")
    for r in feat_rows:
        print(f"  {r[1]}. {r[0]:<8} mean|c| {r[2]:.3f}  median {r[3]:.3f}  max {r[4]:.3f} (P{r[5]:02d})")
    print("Share of total |coef| on A_mean+B_mean:", summary["share_of_total_abs_coef_on_mean_features"])
    print("Per-class share on means:", summary["per_class_share_on_mean_features"])
    print("Max |column sum| (softmax-invariance check):", summary["max_abs_column_sum"])
    for k, v in focus.items():
        print(k, json.dumps(v))
    print(f"Artifacts written to {os.path.relpath(OUT_DIR, lb.REPO_DIR)}")


if __name__ == "__main__":
    main()
