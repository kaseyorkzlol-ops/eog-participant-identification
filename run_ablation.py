"""
Feature-ablation experiments (chronological split + random-split seed sweep).

Question: how much does participant identification depend on the per-window
MEAN features, which carry each recording's DC level (possibly an electrode /
amplifier / session offset), versus the offset-insensitive std/range features?

Feature sets (baseline.FEATURE_SETS):
  all_6          A_mean, A_std, A_range, B_mean, B_std, B_range
  means_only     A_mean, B_mean
  std_range_only A_std, A_range, B_std, B_range

Everything else is identical to the baseline in baseline.py: same windows,
same per-participant 20/7/7 splits, StandardScaler fit on training rows only,
same LogisticRegression settings.

Part A: chronological split (one deterministic split).
Part B: random within-participant split, seeds 0-99. For a given seed, all three
        feature sets use the SAME split, so differences between feature sets
        are paired. Repeated random splits are repeated partitions of the same
        27 recording sessions, not independent replications.

Run:  python run_ablation.py      -> results/ablation/
"""

import json
import os

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import baseline as lb

OUT_DIR = os.path.join(lb.RESULTS_ROOT, "ablation")
SWEEP_SEEDS = list(range(100))       # seed 0 is the originally reported random split
SET_LABELS = {"all_6": "All 6", "means_only": "Means only", "std_range_only": "Std/range only"}


def summarize(values):
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(np.median(v)),
        "min": float(v.min()), "max": float(v.max()),
        "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
    }


def save_confusion_panels(cms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, (name, (cm, acc, f1)) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=7)
        ticks = range(27)
        ax.set_xticks(ticks, [str(p) for p in lb.PARTICIPANT_IDS], fontsize=6)
        ax.set_yticks(ticks, [str(p) for p in lb.PARTICIPANT_IDS], fontsize=6)
        for i in range(27):
            for j in range(27):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                            color="white" if cm[i, j] > 3 else "#333333")
        ax.set_title(f"{SET_LABELS[name]}\ntest accuracy {acc:.3f}, macro-F1 {f1:.3f}", fontsize=10)
        ax.set_xlabel("Predicted participant")
        ax.set_ylabel("True participant")
    fig.suptitle("Chronological split - test confusion matrices by feature set "
                 "(7 test windows per participant)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_distribution_plot(sweep, chrono, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random_color, chrono_color = "#2a78d6", "#eb6834"
    rng = np.random.default_rng(0)   # only for horizontal jitter of the dots
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(lb.FEATURE_SETS)
    for x, name in enumerate(names):
        acc = np.array([r["test_accuracy"] for r in sweep if r["feature_set"] == name])
        ax.boxplot(acc, positions=[x], widths=0.45, showfliers=False,
                   medianprops={"color": "#333333"}, boxprops={"color": "#777777"},
                   whiskerprops={"color": "#777777"}, capprops={"color": "#777777"})
        ax.scatter(x + rng.uniform(-0.17, 0.17, len(acc)), acc, s=10, color=random_color,
                   alpha=0.55, linewidths=0, zorder=3,
                   label="Random within-participant split (seeds 0-99)" if x == 0 else None)
        c = chrono[name]["test_accuracy"]
        ax.scatter([x], [c], marker="D", s=70, color=chrono_color, edgecolors="white",
                   linewidths=1.5, zorder=4, label="Chronological split" if x == 0 else None)
        ax.annotate(f"{c:.3f}", (x + 0.28, c), va="center", fontsize=9, color="#333333")
    ax.axhline(1 / 27, color="#999999", lw=1, ls=":")
    ax.text(len(names) - 0.5, 1 / 27 + 0.01, "chance 1/27", fontsize=8, color="#666666", ha="right")
    ax.set_xticks(range(len(names)), [SET_LABELS[n] for n in names])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Logistic regression test accuracy by feature set", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_all, y, window_index, _, _ = lb.build_dataset()
    lb.check_dataset(X_all, y)

    # ------------------------------------------------------------------
    # Part A: chronological split
    # ------------------------------------------------------------------
    split = lb.make_split(y, window_index, "chronological")
    lb.check_split(split, y, len(y))
    y_test = y[split["test"]]

    chrono, cms, per_class_rows, summary_rows = {}, {}, [], []
    for name, cols in lb.FEATURE_SETS.items():
        X = lb.select_features(X_all, cols)
        res = lb.run_experiment(X, y, split)
        m = res["metrics"]
        chrono[name] = {k: round(float(v), 4) if isinstance(v, float) else v for k, v in m.items()}
        cms[name] = (res["test_confusion"], m["test_accuracy"], m["test_macro_f1"])
        summary_rows.append([SET_LABELS[name], "+".join(cols), round(m["train_accuracy"], 4),
                             round(m["val_accuracy"], 4), round(m["test_accuracy"], 4),
                             round(m["test_macro_f1"], 4)])

        p, r, f, s = precision_recall_fscore_support(
            y_test, res["pred"]["test"], labels=lb.PARTICIPANT_IDS, zero_division=0)
        for i, pid in enumerate(lb.PARTICIPANT_IDS):
            per_class_rows.append([name, pid, round(p[i], 4), round(r[i], 4), round(f[i], 4), int(s[i])])

        lb.write_csv(os.path.join(OUT_DIR, f"confusion_matrix_chronological_{name}.csv"),
                     ["true\\pred"] + lb.PARTICIPANT_IDS,
                     [[pid] + res["test_confusion"][i].tolist() for i, pid in enumerate(lb.PARTICIPANT_IDS)])

    lb.write_csv(os.path.join(OUT_DIR, "ablation_chronological_summary.csv"),
                 ["feature_set", "features", "train_accuracy", "val_accuracy",
                  "test_accuracy", "test_macro_f1"], summary_rows)
    lb.write_csv(os.path.join(OUT_DIR, "ablation_chronological_per_participant.csv"),
                 ["feature_set", "participant", "test_precision", "test_recall", "test_f1",
                  "n_test_windows"], per_class_rows)
    save_confusion_panels(cms, os.path.join(OUT_DIR, "confusion_matrices_chronological_by_feature_set.png"))

    # sanity check: the all-6 run must reproduce the baseline exactly
    with open(os.path.join(lb.RESULTS_DIR, "metrics.json")) as fh:
        base = json.load(fh)["chronological"]
    assert abs(base["test_accuracy"] - chrono["all_6"]["test_accuracy"]) < 1e-4, "baseline not reproduced"

    # ------------------------------------------------------------------
    # Part B: random within-participant split, seeds 0-99
    # ------------------------------------------------------------------
    sweep = []
    for seed in SWEEP_SEEDS:
        split_r = lb.make_split(y, window_index, "random", seed=seed)
        lb.check_split(split_r, y, len(y))
        for name, cols in lb.FEATURE_SETS.items():
            m = lb.run_experiment(lb.select_features(X_all, cols), y, split_r)["metrics"]
            sweep.append({"seed": seed, "feature_set": name,
                          **{k: float(m[k]) for k in ("train_accuracy", "val_accuracy",
                                                      "test_accuracy", "test_macro_f1")}})
    lb.write_csv(os.path.join(OUT_DIR, "random_split_seed_sweep.csv"),
                 ["seed", "feature_set", "train_accuracy", "val_accuracy", "test_accuracy", "test_macro_f1"],
                 [[r["seed"], r["feature_set"], round(r["train_accuracy"], 6), round(r["val_accuracy"], 6),
                   round(r["test_accuracy"], 6), round(r["test_macro_f1"], 6)] for r in sweep])

    # consistency with the baseline's own random-split numbers (seeds 0 and 1-20)
    base_seed0 = base_random = None
    with open(os.path.join(lb.RESULTS_DIR, "metrics.json")) as fh:
        bm = json.load(fh)
        base_seed0 = bm["random"]["test_accuracy"]
        base_random = bm["supplementary_random_split_other_seeds"]["test_accuracy_mean"]
    all6 = {r["seed"]: r for r in sweep if r["feature_set"] == "all_6"}
    assert abs(all6[0]["test_accuracy"] - base_seed0) < 1e-4
    assert abs(np.mean([all6[s]["test_accuracy"] for s in range(1, 21)]) - base_random) < 1e-4

    random_summary, dist_rows = {}, []
    for name in lb.FEATURE_SETS:
        rows = [r for r in sweep if r["feature_set"] == name]
        acc = summarize([r["test_accuracy"] for r in rows])
        f1 = summarize([r["test_macro_f1"] for r in rows])
        seed0 = next(r for r in rows if r["seed"] == 0)
        n_beat = sum(r["test_accuracy"] > chrono[name]["test_accuracy"] for r in rows)
        random_summary[name] = {
            "test_accuracy": {k: round(v, 4) for k, v in acc.items()},
            "test_macro_f1_mean": round(f1["mean"], 4), "test_macro_f1_sd": round(f1["sd"], 4),
            "val_accuracy_mean": round(float(np.mean([r["val_accuracy"] for r in rows])), 4),
            "train_accuracy_mean": round(float(np.mean([r["train_accuracy"] for r in rows])), 4),
            "seed0": {k: round(seed0[k], 4) for k in ("val_accuracy", "test_accuracy", "test_macro_f1")},
            "chronological_test_accuracy": chrono[name]["test_accuracy"],
            "n_seeds_random_above_chronological": int(n_beat),
            "mean_random_minus_chronological_test_accuracy":
                round(acc["mean"] - chrono[name]["test_accuracy"], 4),
        }
        dist_rows.append([SET_LABELS[name], len(rows), *(round(acc[k], 4) for k in
                          ("mean", "sd", "median", "min", "max", "p05", "p95")),
                          round(f1["mean"], 4), round(f1["sd"], 4),
                          round(seed0["test_accuracy"], 4), chrono[name]["test_accuracy"], n_beat])
    lb.write_csv(os.path.join(OUT_DIR, "random_split_seed_sweep_summary.csv"),
                 ["feature_set", "n_seeds", "test_acc_mean", "test_acc_sd", "test_acc_median",
                  "test_acc_min", "test_acc_max", "test_acc_p05", "test_acc_p95",
                  "test_macro_f1_mean", "test_macro_f1_sd", "seed0_test_acc",
                  "chronological_test_acc", "n_seeds_random_above_chronological"], dist_rows)

    # paired comparisons between feature sets on identical random splits
    paired = {}
    for a, b in (("all_6", "std_range_only"), ("all_6", "means_only"), ("means_only", "std_range_only")):
        d = np.array([next(r for r in sweep if r["seed"] == s and r["feature_set"] == a)["test_accuracy"]
                      - next(r for r in sweep if r["seed"] == s and r["feature_set"] == b)["test_accuracy"]
                      for s in SWEEP_SEEDS])
        paired[f"{a}_minus_{b}"] = {"mean": round(float(d.mean()), 4), "sd": round(float(d.std(ddof=1)), 4),
                                    "n_seeds_positive": int((d > 0).sum()),
                                    "n_seeds_zero": int((d == 0).sum())}

    save_distribution_plot(sweep, chrono, os.path.join(OUT_DIR, "random_split_accuracy_distribution.png"))

    summary = {
        "description": "Feature ablation with the baseline logistic-regression pipeline. "
                       "Random-split statistics are over repeated partitions of the same 27 recordings.",
        "feature_sets": lb.FEATURE_SETS,
        "chronological": chrono,
        "random_split_seeds": [SWEEP_SEEDS[0], SWEEP_SEEDS[-1]],
        "random_split": random_summary,
        "paired_random_split_differences_test_accuracy": paired,
    }
    with open(os.path.join(OUT_DIR, "ablation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- console summary ----
    print("CHRONOLOGICAL SPLIT")
    print(f"  {'Feature set':<16}{'Train':>8}{'Val':>8}{'Test':>8}{'MacroF1':>9}")
    for row in summary_rows:
        print(f"  {row[0]:<16}{row[2]:>8.4f}{row[3]:>8.4f}{row[4]:>8.4f}{row[5]:>9.4f}")
    print(f"\nRANDOM SPLIT, seeds {SWEEP_SEEDS[0]}-{SWEEP_SEEDS[-1]} (test accuracy)")
    for name, s in random_summary.items():
        a = s["test_accuracy"]
        print(f"  {SET_LABELS[name]:<16} mean {a['mean']:.4f} sd {a['sd']:.4f} median {a['median']:.4f} "
              f"[{a['min']:.4f}, {a['max']:.4f}] p05-p95 [{a['p05']:.4f}, {a['p95']:.4f}]  "
              f"F1 {s['test_macro_f1_mean']:.4f}  seed0 {s['seed0']['test_accuracy']:.4f}  "
              f"> chrono in {s['n_seeds_random_above_chronological']}/100")
    print("\nPaired differences:", json.dumps(paired))
    print(f"\nArtifacts written to {os.path.relpath(OUT_DIR, lb.REPO_DIR)}")


if __name__ == "__main__":
    main()
