"""
Descriptive quantification of within-recording temporal drift.

For every participant and every one of the six features, fit an ordinary
least-squares line  feature ~ intercept + slope * window_index  over that
participant's 34 sequential 2-s windows. A straight line is only a simple
descriptive summary; it does not claim the drift is linear.

Saved per participant x feature:
  slope_per_window, slope_per_minute, intercept (value at window 0), r2,
  fitted_change      = slope * (n_windows - 1)   (start-to-end change of the fitted line)
  observed_change    = last window value - first window value
  normalized_change  = fitted_change / between_participant_sd
  robust_change      = median(last 11 windows) - median(first 11 windows)
  robust_normalized_change = robust_change / between_participant_sd
where between_participant_sd is the SD (across the 27 participants) of the
per-participant mean of that feature. It makes features with different units
comparable: |normalized_change| = 1 means the fitted drift within ONE recording
is as large as the typical spread of that feature BETWEEN participants.

Caveat: for std/range features an isolated spike window (e.g. a blink burst)
can tilt the OLS line even when there is no gradual trend (R^2 stays low).
robust_change (thirds, medians) is insensitive to such spikes, and R^2 says
how much of a feature's within-recording variation a straight line explains.

Run:  python run_drift_analysis.py      -> results/drift/
"""

import json
import os

import numpy as np

import baseline as lb

OUT_DIR = os.path.join(lb.RESULTS_ROOT, "drift")
REPRESENTATIVE = [1, 12, 13, 25]   # participants selected for plotting after the baseline analysis


def fit_line(t, v):
    """OLS fit v = intercept + slope * t. Returns slope, intercept, R^2."""
    slope, intercept = np.polyfit(t, v, 1)
    resid = v - (intercept + slope * t)
    ss_tot = np.sum((v - v.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, window_index, _, _ = lb.build_dataset()
    F = lb.FEATURE_NAMES
    pids = lb.PARTICIPANT_IDS

    # per-participant means -> between-participant SD for each feature
    part_means = np.array([[X[y == p, j].mean() for j in range(len(F))] for p in pids])
    between_sd = part_means.std(axis=0, ddof=1)

    rows = []
    norm = np.zeros((len(pids), len(F)))
    rnorm = np.zeros((len(pids), len(F)))
    r2_mat = np.zeros((len(pids), len(F)))
    for i, pid in enumerate(pids):
        idx = np.where(y == pid)[0]
        idx = idx[np.argsort(window_index[idx])]
        t = window_index[idx].astype(float)
        for j, name in enumerate(F):
            v = X[idx, j]
            slope, intercept, r2 = fit_line(t, v)
            fitted_change = slope * (len(t) - 1)
            third = len(t) // 3
            robust_change = float(np.median(v[-third:]) - np.median(v[:third]))
            norm[i, j] = fitted_change / between_sd[j]
            rnorm[i, j] = robust_change / between_sd[j]
            r2_mat[i, j] = r2
            rows.append({
                "participant": pid, "feature": name, "n_windows": len(t),
                "slope_per_window": slope, "slope_per_minute": slope * 30.0,   # 30 windows of 2 s per minute
                "intercept": intercept, "r2": r2,
                "fitted_change": fitted_change, "observed_change": float(v[-1] - v[0]),
                "participant_mean": float(v.mean()), "within_participant_sd": float(v.std(ddof=1)),
                "between_participant_sd": float(between_sd[j]), "normalized_change": float(norm[i, j]),
                "robust_change": robust_change, "robust_normalized_change": float(rnorm[i, j]),
            })
    cols = list(rows[0])
    lb.write_csv(os.path.join(OUT_DIR, "drift_linear_trends.csv"), cols,
                 [[round(r[c], 6) if isinstance(r[c], float) else r[c] for c in cols] for r in rows])

    # ---- per-feature summary ----
    feat_summary = []
    for j, name in enumerate(F):
        rr = [r for r in rows if r["feature"] == name]
        an = np.abs(norm[:, j])
        feat_summary.append({
            "feature": name,
            "median_r2": float(np.median(r2_mat[:, j])),
            "n_participants_r2_above_0.5": int((r2_mat[:, j] > 0.5).sum()),
            "median_abs_normalized_change": float(np.median(an)),
            "max_abs_normalized_change": float(an.max()),
            "n_participants_abs_normalized_change_above_0.5": int((an > 0.5).sum()),
            "median_abs_robust_normalized_change": float(np.median(np.abs(rnorm[:, j]))),
            "n_participants_abs_robust_normalized_change_above_0.5": int((np.abs(rnorm[:, j]) > 0.5).sum()),
            "median_within_over_between_sd": float(np.median(
                [r["within_participant_sd"] for r in rr]) / between_sd[j]),
            "median_slope_per_minute": float(np.median([r["slope_per_minute"] for r in rr])),
            "median_abs_slope_per_minute": float(np.median([abs(r["slope_per_minute"]) for r in rr])),
            "n_positive_slopes": int(sum(r["slope_per_window"] > 0 for r in rr)),
            "between_participant_sd": float(between_sd[j]),
        })
    fcols = list(feat_summary[0])
    lb.write_csv(os.path.join(OUT_DIR, "drift_summary_by_feature.csv"), fcols,
                 [[round(s[c], 4) if isinstance(s[c], float) else s[c] for c in fcols] for s in feat_summary])

    # ---- largest drifts ----
    largest = {}
    for j, name in enumerate(F):
        order = np.argsort(-np.abs(norm[:, j]))[:5]
        largest[name] = [{"participant": pids[i], "fitted_change": round(norm[i, j] * between_sd[j], 4),
                          "normalized_change": round(float(norm[i, j]), 3),
                          "robust_normalized_change": round(float(rnorm[i, j]), 3),
                          "r2": round(float(r2_mat[i, j]), 3)} for i in order]
    flat = sorted(rows, key=lambda r: -abs(r["robust_normalized_change"]))[:10]
    largest_overall = [{k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in
                        ("participant", "feature", "fitted_change", "observed_change", "robust_change",
                         "normalized_change", "robust_normalized_change", "r2")} for r in flat]

    p01 = next(r for r in rows if r["participant"] == 1 and r["feature"] == "A_mean")
    summary = {
        "method": "OLS line per participant x feature over window index (descriptive only)",
        "by_feature": feat_summary,
        "largest_abs_normalized_change_per_feature_top5": largest,
        "largest_abs_robust_normalized_change_overall_top10": largest_overall,
        "P01_A_mean_check": {k: round(p01[k], 4) if isinstance(p01[k], float) else p01[k]
                             for k in ("slope_per_minute", "intercept", "r2", "fitted_change",
                                       "observed_change", "robust_change", "normalized_change")},
    }
    with open(os.path.join(OUT_DIR, "drift_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- figure 1: participant x feature heatmaps (R^2 and signed normalized change) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 9))
    im0 = axes[0].imshow(r2_mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    lim = np.ceil(np.abs(rnorm).max())
    im1 = axes[1].imshow(rnorm, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for ax, mat, fmt, title in ((axes[0], r2_mat, "{:.2f}", "R² of linear trend"),
                                (axes[1], rnorm, "{:+.1f}", "Robust change: median(last third) - median(first third)\n"
                                 "/ between-participant SD of feature")):
        ax.set_xticks(range(len(F)), F, rotation=30, fontsize=8)
        ax.set_yticks(range(len(pids)), [f"P{p:02d}" for p in pids], fontsize=7)
        ax.set_title(title, fontsize=10)
        for i in range(len(pids)):
            for j in range(len(F)):
                val = mat[i, j]
                dark = val > 0.6 if mat is r2_mat else abs(val) > lim * 0.55
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=6,
                        color="white" if dark else "#333333")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    fig.suptitle("Within-recording drift per participant and feature (34 windows, ~68 s)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "drift_heatmap_participant_by_feature.png"), dpi=150)
    plt.close(fig)

    # ---- figure 2: distribution across participants, per feature ----
    rng = np.random.default_rng(0)   # jitter only
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, ylabel in ((axes[0], r2_mat, "R² of linear trend"),
                            (axes[1], np.abs(rnorm), "|robust change| / between-participant SD")):
        for j in range(len(F)):
            ax.boxplot(mat[:, j], positions=[j], widths=0.5, showfliers=False,
                       medianprops={"color": "#333333"}, boxprops={"color": "#888888"},
                       whiskerprops={"color": "#888888"}, capprops={"color": "#888888"})
            ax.scatter(j + rng.uniform(-0.15, 0.15, len(pids)), mat[:, j], s=12,
                       color="#2a78d6", alpha=0.6, linewidths=0, zorder=3)
        ax.set_xticks(range(len(F)), F, rotation=30, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[1].set_yscale("log")
    fig.suptitle("Drift across the 27 participants (one dot per participant)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "drift_distribution_by_feature.png"), dpi=150)
    plt.close(fig)

    # ---- figure 3: representative participants, raw feature values with fitted lines ----
    reps = REPRESENTATIVE + [pids[int(np.argmin(np.abs(rnorm).max(axis=1)))]]   # + least-drifting participant
    fig, axes = plt.subplots(len(reps), len(F), figsize=(15, 2.2 * len(reps)), sharex=True)
    n_tr, n_va, _ = lb.split_sizes(34)
    for r_i, pid in enumerate(reps):
        idx = np.where(y == pid)[0]
        idx = idx[np.argsort(window_index[idx])]
        t = window_index[idx]
        for j, name in enumerate(F):
            ax = axes[r_i, j]
            v = X[idx, j]
            rec = next(r for r in rows if r["participant"] == pid and r["feature"] == name)
            ax.plot(t, v, color="#2a78d6", lw=1.2, marker="o", ms=2.5)
            ax.plot(t, rec["intercept"] + rec["slope_per_window"] * t, color="#eb6834", lw=1.5)
            for b in (n_tr - 0.5, n_tr + n_va - 0.5):
                ax.axvline(b, color="#999999", lw=0.8, ls="--")
            ax.text(0.02, 0.92, f"R²={rec['r2']:.2f}", transform=ax.transAxes, fontsize=7,
                    va="top", color="#333333")
            ax.tick_params(labelsize=7)
            if r_i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"P{pid:02d}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("window index", fontsize=8)
    fig.suptitle("Representative participants: feature per 2-s window (blue) and linear trend (orange); "
                 "dashed = chronological train | val | test boundaries.  Last row = least-drifting participant",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "drift_representative_participants.png"), dpi=130)
    plt.close(fig)

    # ---- console ----
    print(f"{'feature':<9}{'median R2':>10}{'#R2>0.5':>9}{'med|norm|':>11}{'med|robust|':>13}"
          f"{'#|robust|>0.5':>15}{'within/between SD':>19}")
    for s in feat_summary:
        print(f"{s['feature']:<9}{s['median_r2']:>10.3f}{s['n_participants_r2_above_0.5']:>9}"
              f"{s['median_abs_normalized_change']:>11.3f}{s['median_abs_robust_normalized_change']:>13.3f}"
              f"{s['n_participants_abs_robust_normalized_change_above_0.5']:>15}"
              f"{s['median_within_over_between_sd']:>19.3f}")
    print("\nP01 A_mean:", summary["P01_A_mean_check"])
    print("\nTop 10 overall:")
    for r in largest_overall:
        print("  ", r)
    print(f"\nRepresentative participants plotted: {reps}")
    print(f"Artifacts written to {os.path.relpath(OUT_DIR, lb.REPO_DIR)}")


if __name__ == "__main__":
    main()
