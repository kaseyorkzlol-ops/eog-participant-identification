"""
Modest model comparison: do the main conclusions depend on logistic regression?

Six classifiers with fixed, documented parameters (scikit-learn defaults unless
stated). NO hyperparameter tuning is done: nothing is selected using
validation or test data. Logistic regression remains the primary baseline.

Every model gets the same input: StandardScaler fit on the TRAINING rows only,
applied to val/test. (Scaling is required for SVM/kNN/LR, is irrelevant for LDA's
predictions and for random forests, and is applied uniformly for simplicity.)

Feature sets: all_6 (primary), std_range_only (key diagnostic), means_only.
Splits:
  - chronological 20/7/7 (primary)
  - random within-participant split, seeds 0-19 (mean reported), to check
    whether "random split is more optimistic than chronological" holds beyond LR.

Run:  python run_model_comparison.py      -> results/model_comparison/
"""

import json
import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import baseline as lb

OUT_DIR = os.path.join(lb.RESULTS_ROOT, "model_comparison")
RANDOM_SEEDS = list(range(20))

# name -> (factory, parameter description written to the outputs)
MODELS = {
    "LogisticRegression": (lambda: LogisticRegression(max_iter=10000, random_state=lb.SEED),
                           "C=1.0, L2, lbfgs, max_iter=10000 (identical to baseline)"),
    "LDA": (lambda: LinearDiscriminantAnalysis(), "solver=svd, no shrinkage (defaults)"),
    "LinearSVM": (lambda: SVC(kernel="linear", C=1.0, random_state=lb.SEED),
                  "SVC kernel=linear, C=1.0, one-vs-one"),
    "RBF_SVM": (lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=lb.SEED),
                "SVC kernel=rbf, C=1.0, gamma='scale', one-vs-one"),
    "RandomForest": (lambda: RandomForestClassifier(n_estimators=500, random_state=lb.SEED, n_jobs=1),
                     "n_estimators=500, other parameters default, random_state=0"),
    "kNN": (lambda: KNeighborsClassifier(n_neighbors=5), "k=5, uniform weights, Euclidean (defaults)"),
}


def evaluate(model_name, X, y, split):
    """Fit scaler on train only, fit model, return accuracy/F1 on all splits."""
    scaler = StandardScaler().fit(X[split["train"]])
    assert scaler.n_samples_seen_ == len(split["train"])
    model = MODELS[model_name][0]()
    model.fit(scaler.transform(X[split["train"]]), y[split["train"]])
    out = {}
    for k in ("train", "val", "test"):
        pred = model.predict(scaler.transform(X[split[k]]))
        out[f"{k}_accuracy"] = accuracy_score(y[split[k]], pred)
        out[f"{k}_macro_f1"] = f1_score(y[split[k]], pred, average="macro")
    return out


def save_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(MODELS)
    sets = list(lb.FEATURE_SETS)
    fig, axes = plt.subplots(1, len(sets), figsize=(14, 4.6), sharey=True)
    for ax, fs in zip(axes, sets):
        for i, m in enumerate(names):
            r = next(r for r in rows if r["model"] == m and r["feature_set"] == fs)
            ax.plot([r["chrono_test_accuracy"], r["random_test_accuracy_mean"]], [i, i],
                    color="#bbbbbb", lw=1.5, zorder=1)
            ax.scatter(r["chrono_test_accuracy"], i, color="#eb6834", s=55, zorder=3,
                       label="Chronological test acc." if i == 0 else None)
            ax.scatter(r["random_test_accuracy_mean"], i, color="#2a78d6", s=55, zorder=3,
                       label="Random split, mean of seeds 0-19" if i == 0 else None)
        ax.axvline(1 / 27, color="#999999", lw=1, ls=":")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Test accuracy")
        ax.set_title({"all_6": "All 6 features", "means_only": "Means only",
                      "std_range_only": "Std/range only"}[fs], fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis() if fs == sets[0] else None
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Model comparison (fixed parameters, no tuning); dotted line = chance 1/27", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_all, y, window_index, _, _ = lb.build_dataset()
    chrono_split = lb.make_split(y, window_index, "chronological")
    lb.check_split(chrono_split, y, len(y))
    random_splits = {s: lb.make_split(y, window_index, "random", seed=s) for s in RANDOM_SEEDS}

    rows = []
    for fs, cols in lb.FEATURE_SETS.items():
        X = lb.select_features(X_all, cols)
        for m in MODELS:
            c = evaluate(m, X, y, chrono_split)
            rand = [evaluate(m, X, y, random_splits[s]) for s in RANDOM_SEEDS]
            racc = np.array([r["test_accuracy"] for r in rand])
            rows.append({
                "feature_set": fs, "model": m, "parameters": MODELS[m][1],
                "chrono_train_accuracy": c["train_accuracy"], "chrono_val_accuracy": c["val_accuracy"],
                "chrono_test_accuracy": c["test_accuracy"], "chrono_test_macro_f1": c["test_macro_f1"],
                "random_test_accuracy_mean": float(racc.mean()), "random_test_accuracy_sd": float(racc.std(ddof=1)),
                "random_test_macro_f1_mean": float(np.mean([r["test_macro_f1"] for r in rand])),
                "random_minus_chrono_test_accuracy": float(racc.mean() - c["test_accuracy"]),
            })
            r = rows[-1]
            print(f"{fs:<15}{m:<19} chrono train {r['chrono_train_accuracy']:.3f} val {r['chrono_val_accuracy']:.3f} "
                  f"test {r['chrono_test_accuracy']:.3f} F1 {r['chrono_test_macro_f1']:.3f} | random mean "
                  f"{r['random_test_accuracy_mean']:.3f} (sd {r['random_test_accuracy_sd']:.3f})")

    # LR row must match the baseline exactly
    lr = next(r for r in rows if r["model"] == "LogisticRegression" and r["feature_set"] == "all_6")
    assert abs(lr["chrono_test_accuracy"] - 0.5450) < 1e-3

    cols = list(rows[0])
    lb.write_csv(os.path.join(OUT_DIR, "model_comparison.csv"), cols,
                 [[round(r[c], 4) if isinstance(r[c], float) else r[c] for c in cols] for r in rows])
    with open(os.path.join(OUT_DIR, "model_comparison_summary.json"), "w") as fh:
        json.dump({"note": "Fixed parameters, no tuning. Random-split values are means over seeds "
                           f"{RANDOM_SEEDS[0]}-{RANDOM_SEEDS[-1]} (repeated partitions of the same recordings).",
                   "rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]},
                  fh, indent=2)
    save_plot(rows, os.path.join(OUT_DIR, "model_comparison_test_accuracy.png"))
    print(f"Artifacts written to {os.path.relpath(OUT_DIR, lb.REPO_DIR)}")


if __name__ == "__main__":
    main()
