"""
EOG participant-identification baseline (closed-set, 27 identities).

Pipeline
  1. Data loading       - read A (horizontal) and B (vertical) EOG from each .mat file
  2. Window extraction  - non-overlapping 2 s windows (500 samples @ 250 Hz)
  3. Feature extraction - 6 features per window: mean / std / range of A and B
  4. Splitting          - per participant, 60/20/20 train/val/test
                          Experiment 1: chronological (no shuffling)
                          Experiment 2: windows shuffled WITHIN each participant (fixed seed)
  5. Preprocessing+model- StandardScaler fit on TRAIN ONLY -> LogisticRegression
  6. Evaluation         - accuracy, macro-F1, classification report, 27x27 confusion matrix

The two experiments use exactly the same windows, features, scaler and model;
only the assignment of windows to train/val/test differs.

Run (paths are resolved relative to this file, so any working directory works):
    python baseline.py
Outputs are written to results/baseline/.

The loading and feature code is adapted from the original inspect_eog.py.
Other scripts (run_ablation.py, run_drift_analysis.py, ...) import the
functions defined here so every experiment uses the same windows and features.
"""

import csv
import json
import os

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(REPO_DIR, "Data", "EOG Data", "Raw Data")
RESULTS_ROOT = os.path.join(REPO_DIR, "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, "baseline")

FS = 250                      # sampling rate (Hz)
WINDOW_SECONDS = 2
WINDOW_SIZE = FS * WINDOW_SECONDS   # 500 samples, windows do NOT overlap
PARTICIPANT_IDS = list(range(1, 28))

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2                # test gets the remainder (~0.2)

SEED = 0                      # used for the random-window split and the model
EXTRA_RANDOM_SEEDS = list(range(1, 21))  # supplementary: random-split variability only

FEATURE_NAMES = ["A_mean", "A_std", "A_range", "B_mean", "B_std", "B_range"]

# Feature subsets used by the ablation experiments (run_ablation.py, run_model_comparison.py).
# The mean features carry each recording's DC level, which may reflect
# electrode/amplifier/session offsets; std and range do not depend on that level.
FEATURE_SETS = {
    "all_6": FEATURE_NAMES,
    "means_only": ["A_mean", "B_mean"],
    "std_range_only": ["A_std", "A_range", "B_std", "B_range"],
}


def select_features(X, names):
    """Return the columns of X named in `names` (in that order)."""
    cols = [FEATURE_NAMES.index(n) for n in names]
    return X[:, cols]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_participant_signals(participant_id):
    """Return (A, B, mat_format) for one participant as 1-D float arrays.

    Files are named 01.mat ... 27.mat. Most are MATLAB v5 (scipy.io.loadmat);
    15.mat is MATLAB v7.3 (HDF5), which loadmat refuses, so we fall back to h5py.
    The raw files are only read, never modified.
    """
    filename = os.path.join(RAW_DIR, f"{participant_id:02d}.mat")
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Raw EOG file not found: {os.path.relpath(filename, REPO_DIR)}\n"
            "The dataset is not included in this repository. Obtain it separately "
            "(see README.md, 'Getting the data') and place the 27 files at "
            "Data/EOG Data/Raw Data/01.mat ... 27.mat inside this repository folder.")
    try:
        data = loadmat(filename)
        A = data["A"].astype(float).ravel()
        B = data["B"].astype(float).ravel()
        mat_format = "v5 (scipy.io.loadmat)"
    except NotImplementedError:
        with h5py.File(filename, "r") as data:
            A = np.array(data["A"], dtype=float).ravel()
            B = np.array(data["B"], dtype=float).ravel()
        mat_format = "v7.3 (h5py)"

    assert len(A) == len(B), f"P{participant_id}: A and B have different lengths"
    return A, B, mat_format


# ---------------------------------------------------------------------------
# 2. Window extraction
# ---------------------------------------------------------------------------
def extract_windows(A, B, window_size=WINDOW_SIZE):
    """Cut A and B into consecutive NON-overlapping windows.

    Window k covers samples [k*window_size, (k+1)*window_size). Leftover samples
    at the end that do not fill a whole window are discarded.
    Returns (A_windows, B_windows, start_samples), each with one row per window,
    in chronological order.
    """
    n_windows = min(len(A), len(B)) // window_size
    starts = np.arange(n_windows) * window_size
    A_windows = np.stack([A[s:s + window_size] for s in starts])
    B_windows = np.stack([B[s:s + window_size] for s in starts])
    return A_windows, B_windows, starts


# ---------------------------------------------------------------------------
# 3. Feature extraction
# ---------------------------------------------------------------------------
def window_features(a, b):
    """The six baseline features for one window (order matches FEATURE_NAMES).

    std is the population standard deviation (numpy default, ddof=0).
    range = max - min.
    """
    return [
        np.mean(a), np.std(a), np.max(a) - np.min(a),
        np.mean(b), np.std(b), np.max(b) - np.min(b),
    ]


def build_dataset():
    """Load all participants and build X (n_windows x 6), y, and bookkeeping arrays.

    Rows are ordered by participant, then chronologically by window index.
    """
    X_rows, y, window_index, start_sample = [], [], [], []
    info = []
    for pid in PARTICIPANT_IDS:
        A, B, mat_format = load_participant_signals(pid)
        A_win, B_win, starts = extract_windows(A, B)
        for k in range(len(starts)):
            X_rows.append(window_features(A_win[k], B_win[k]))
            y.append(pid)
            window_index.append(k)
            start_sample.append(int(starts[k]))
        info.append({
            "participant": pid,
            "mat_format": mat_format,
            "n_samples": len(A),
            "duration_s": round(len(A) / FS, 3),
            "n_windows": len(starts),
            "discarded_tail_samples": len(A) - len(starts) * WINDOW_SIZE,
        })
    return (np.array(X_rows, dtype=float), np.array(y), np.array(window_index),
            np.array(start_sample), info)


def check_dataset(X, y):
    """Sanity checks on the feature matrix. Returns a dict of diagnostics."""
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    stds = X.std(axis=0)
    constant = [FEATURE_NAMES[j] for j in range(X.shape[1]) if stds[j] < 1e-12]
    # std/range of a real EOG window should be strictly positive
    nonpositive_spread = {
        name: int((X[:, j] <= 0).sum())
        for j, name in enumerate(FEATURE_NAMES) if name.endswith(("_std", "_range"))
    }
    # flat windows within a participant (e.g. dropped signal) would show std == 0
    assert len(np.unique(y)) == 27, f"expected 27 identities, found {len(np.unique(y))}"
    assert n_nan == 0 and n_inf == 0, "feature matrix contains NaN/Inf"
    assert not constant, f"constant features: {constant}"
    return {
        "n_nan": n_nan, "n_inf": n_inf,
        "constant_features": constant,
        "nonpositive_std_or_range_counts": nonpositive_spread,
        "feature_min": dict(zip(FEATURE_NAMES, X.min(axis=0).round(6).tolist())),
        "feature_max": dict(zip(FEATURE_NAMES, X.max(axis=0).round(6).tolist())),
    }


# ---------------------------------------------------------------------------
# 4. Splitting (always done separately for each participant)
# ---------------------------------------------------------------------------
def split_sizes(n):
    """Number of train / val / test windows for a participant with n windows."""
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * (TRAIN_FRAC + VAL_FRAC)) - n_train
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def make_split(y, window_index, mode, seed=SEED):
    """Return dict of row indices {'train','val','test'} into X.

    mode == 'chronological': within each participant, the first 60% of windows
        (in time order) go to train, the next 20% to val, the last 20% to test.
        No shuffling.
    mode == 'random': within each participant, the windows are shuffled with a
        fixed-seed RNG and then cut with the same 60/20/20 sizes. Participants
        are never mixed or held out: every identity appears in every split.
    """
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for pid in PARTICIPANT_IDS:
        rows = np.where(y == pid)[0]
        rows = rows[np.argsort(window_index[rows])]   # chronological order
        if mode == "random":
            rows = rng.permutation(rows)
        elif mode != "chronological":
            raise ValueError(mode)
        n_train, n_val, _ = split_sizes(len(rows))
        split["train"].extend(rows[:n_train])
        split["val"].extend(rows[n_train:n_train + n_val])
        split["test"].extend(rows[n_train + n_val:])
    return {k: np.array(v) for k, v in split.items()}


def check_split(split, y, n_rows):
    """No window in two splits, every window used, every identity in every split."""
    tr, va, te = (set(split[k].tolist()) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te), "overlapping splits"
    assert len(tr) + len(va) + len(te) == n_rows, "some windows not assigned"
    for k in ("train", "val", "test"):
        missing = set(PARTICIPANT_IDS) - set(y[split[k]].tolist())
        assert not missing, f"participants missing from {k}: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 5. Preprocessing + model
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    """Fit StandardScaler on TRAINING rows only, then a multinomial logistic regression.

    Default hyperparameters (C=1.0, L2, lbfgs). No tuning is done. max_iter is
    raised only so the solver converges.
    """
    scaler = StandardScaler().fit(X_train)
    # leakage guard: the scaler's statistics must be exactly those of X_train
    assert scaler.n_samples_seen_ == len(X_train)
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
    assert np.allclose(scaler.scale_, X_train.std(axis=0))

    model = LogisticRegression(max_iter=10000, random_state=SEED)
    model.fit(scaler.transform(X_train), y_train)
    return scaler, model


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
def run_experiment(X, y, split):
    """Train on split['train'], evaluate on all three splits."""
    scaler, model = train_model(X[split["train"]], y[split["train"]])
    out = {"scaler": scaler, "model": model, "pred": {}}
    metrics = {}
    for k in ("train", "val", "test"):
        # val/test are only TRANSFORMED with the scaler fit on train
        pred = model.predict(scaler.transform(X[split[k]]))
        out["pred"][k] = pred
        metrics[f"{k}_accuracy"] = accuracy_score(y[split[k]], pred)
        metrics[f"{k}_macro_f1"] = f1_score(y[split[k]], pred, average="macro")
        metrics[f"n_{k}"] = len(split[k])
    out["metrics"] = metrics
    out["test_confusion"] = confusion_matrix(y[split["test"]], out["pred"]["test"],
                                             labels=PARTICIPANT_IDS)
    out["test_report"] = classification_report(
        y[split["test"]], out["pred"]["test"], labels=PARTICIPANT_IDS,
        target_names=[f"P{p:02d}" for p in PARTICIPANT_IDS], digits=3, zero_division=0)
    return out


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_confusion_png(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
    ax.set_xticks(range(27), [str(p) for p in PARTICIPANT_IDS], fontsize=7)
    ax.set_yticks(range(27), [str(p) for p in PARTICIPANT_IDS], fontsize=7)
    ax.set_xlabel("Predicted participant")
    ax.set_ylabel("True participant")
    ax.set_title(title, fontsize=10)
    for i in range(27):
        for j in range(27):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 2 else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="test windows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_drift_png(X, y, window_index, path):
    """Diagnostic: each feature vs. window index for every participant, with the
    chronological train/val/test boundaries marked. Each participant's curve is
    centred on its own mean so within-recording change over time is visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_train, n_val, _ = split_sizes(int(np.bincount(y).max()))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        for pid in PARTICIPANT_IDS:
            rows = np.where(y == pid)[0]
            rows = rows[np.argsort(window_index[rows])]
            v = X[rows, j]
            ax.plot(window_index[rows], v - v.mean(), color="#2a6fdb", alpha=0.35, lw=1)
        for b in (n_train - 0.5, n_train + n_val - 0.5):
            ax.axvline(b, color="#444444", lw=1, ls="--")
        ax.set_title(FEATURE_NAMES[j] + "  (minus participant mean)", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("window index (2 s each)")
    fig.suptitle("Per-participant feature trajectories over the recording "
                 "(dashed lines = chronological train | val | test boundaries)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- dataset ----
    X, y, window_index, start_sample, info = build_dataset()
    diag = check_dataset(X, y)

    print("=" * 70)
    print("DATASET")
    print("X shape:", X.shape, "  y shape:", y.shape)
    print("Unique participants:", len(np.unique(y)))
    print("Feature names:", FEATURE_NAMES)
    print("NaN count:", diag["n_nan"], " Inf count:", diag["n_inf"])
    print("Constant features:", diag["constant_features"] or "none")
    print("Non-positive std/range counts:", diag["nonpositive_std_or_range_counts"])
    print("\nParticipant  format                  samples  windows  discarded_tail")
    for r in info:
        print(f"  P{r['participant']:02d}       {r['mat_format']:<22}  {r['n_samples']:>6}  "
              f"{r['n_windows']:>6}   {r['discarded_tail_samples']:>5}")

    write_csv(os.path.join(RESULTS_DIR, "window_counts.csv"),
              ["participant", "mat_format", "n_samples", "duration_s", "n_windows",
               "discarded_tail_samples", "n_train", "n_val", "n_test"],
              [[r["participant"], r["mat_format"], r["n_samples"], r["duration_s"],
                r["n_windows"], r["discarded_tail_samples"], *split_sizes(r["n_windows"])]
               for r in info])
    write_csv(os.path.join(RESULTS_DIR, "features.csv"),
              ["row", "participant", "window_index", "start_sample"] + FEATURE_NAMES,
              [[i, int(y[i]), int(window_index[i]), int(start_sample[i])] + X[i].tolist()
               for i in range(len(y))])

    # ---- experiments ----
    all_metrics = {
        "config": {"fs": FS, "window_size": WINDOW_SIZE, "overlap": 0,
                   "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC, "seed": SEED,
                   "model": "StandardScaler(fit on train) + LogisticRegression"
                            "(C=1.0, lbfgs, max_iter=10000)",
                   "features": FEATURE_NAMES},
        "dataset": {"X_shape": list(X.shape), "y_shape": list(y.shape),
                    "n_participants": int(len(np.unique(y))), **diag},
    }
    split_label = {}
    results = {}

    for mode in ("chronological", "random"):
        split = make_split(y, window_index, mode, seed=SEED)
        check_split(split, y, len(y))
        res = run_experiment(X, y, split)
        results[mode] = res
        all_metrics[mode] = {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in res["metrics"].items()}

        # per-window split assignment
        split_label[mode] = np.empty(len(y), dtype=object)
        for k in ("train", "val", "test"):
            split_label[mode][split[k]] = k

        # predictions (val + test)
        pred_rows = []
        for k in ("train", "val", "test"):
            for i, p in zip(split[k], res["pred"][k]):
                pred_rows.append([k, int(i), int(y[i]), int(window_index[i]), int(p), int(p == y[i])])
        write_csv(os.path.join(RESULTS_DIR, f"predictions_{mode}.csv"),
                  ["split", "row", "true_participant", "window_index",
                   "predicted_participant", "correct"], pred_rows)

        cm = res["test_confusion"]
        write_csv(os.path.join(RESULTS_DIR, f"confusion_matrix_{mode}.csv"),
                  ["true\\pred"] + PARTICIPANT_IDS,
                  [[pid] + cm[i].tolist() for i, pid in enumerate(PARTICIPANT_IDS)])
        with open(os.path.join(RESULTS_DIR, f"classification_report_{mode}.txt"), "w") as f:
            f.write(f"{mode} split - TEST set\n\n{res['test_report']}")
        m = res["metrics"]
        save_confusion_png(cm, f"{mode.capitalize()} split - test confusion matrix\n"
                               f"accuracy {m['test_accuracy']:.3f}, macro-F1 {m['test_macro_f1']:.3f}",
                           os.path.join(RESULTS_DIR, f"confusion_matrix_{mode}.png"))

        print("=" * 70)
        print(f"EXPERIMENT: {mode.upper()} SPLIT  "
              f"(train/val/test = {m['n_train']}/{m['n_val']}/{m['n_test']})")
        for k in ("train", "val", "test"):
            print(f"  {k:<5} accuracy {m[f'{k}_accuracy']:.4f}   macro-F1 {m[f'{k}_macro_f1']:.4f}")
        print(res["test_report"])

    write_csv(os.path.join(RESULTS_DIR, "split_assignments.csv"),
              ["row", "participant", "window_index", "chronological_split", "random_split"],
              [[i, int(y[i]), int(window_index[i]), split_label["chronological"][i],
                split_label["random"][i]] for i in range(len(y))])

    # sanity: both experiments really used the same rows and the same sizes
    for k in ("train", "val", "test"):
        assert all_metrics["chronological"][f"n_{k}"] == all_metrics["random"][f"n_{k}"]

    # ---- difference ----
    diff = {f"{k}_{s}": round(all_metrics["random"][f"{k}_{s}"] - all_metrics["chronological"][f"{k}_{s}"], 4)
            for k in ("train", "val", "test") for s in ("accuracy", "macro_f1")}
    all_metrics["difference_random_minus_chronological"] = diff

    # ---- supplementary: how much does the random split vary with the seed? ----
    # Same protocol, only the shuffling seed changes. Not used for any selection.
    extra = []
    for s in EXTRA_RANDOM_SEEDS:
        split = make_split(y, window_index, "random", seed=s)
        check_split(split, y, len(y))
        m = run_experiment(X, y, split)["metrics"]
        extra.append([s, m["val_accuracy"], m["test_accuracy"], m["test_macro_f1"]])
    write_csv(os.path.join(RESULTS_DIR, "random_split_seed_variability.csv"),
              ["seed", "val_accuracy", "test_accuracy", "test_macro_f1"], extra)
    ex = np.array(extra)[:, 1:]
    all_metrics["supplementary_random_split_other_seeds"] = {
        "seeds": EXTRA_RANDOM_SEEDS,
        "test_accuracy_mean": round(float(ex[:, 1].mean()), 4),
        "test_accuracy_std": round(float(ex[:, 1].std()), 4),
        "test_accuracy_min": round(float(ex[:, 1].min()), 4),
        "test_accuracy_max": round(float(ex[:, 1].max()), 4),
        "test_macro_f1_mean": round(float(ex[:, 2].mean()), 4),
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    save_drift_png(X, y, window_index, os.path.join(RESULTS_DIR, "feature_drift_over_time.png"))

    print("=" * 70)
    print("SUMMARY (test set)")
    for mode in ("chronological", "random"):
        m = all_metrics[mode]
        print(f"  {mode:<14} val acc {m['val_accuracy']:.4f}  test acc {m['test_accuracy']:.4f}  "
              f"test macro-F1 {m['test_macro_f1']:.4f}")
    print("  random - chronological:", diff)
    print("  random split, seeds 1-20: test acc mean %.4f (sd %.4f, range %.4f-%.4f)" % (
        ex[:, 1].mean(), ex[:, 1].std(), ex[:, 1].min(), ex[:, 1].max()))
    print(f"\nArtifacts written to {os.path.relpath(RESULTS_DIR, REPO_DIR)}")


if __name__ == "__main__":
    main()
