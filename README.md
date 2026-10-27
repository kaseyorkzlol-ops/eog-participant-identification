# EOG Participant Identification (Within-Session)

Analysis code for **within-session participant identification** from two-channel electrooculography (EOG). The question: given a 2-second window of one participant's recording, how well can simple classifiers tell which of 27 participants it came from, and what signal are they using?

> **Important scope limitation.** The dataset used has **one recording session per participant**. Training and test windows therefore always come from the same recording. These experiments **cannot establish cross-session or biometric identification**; any recording-specific property (electrode placement, skin contact, amplifier offset) is confounded with identity.

> **Status:** Active research project and open-source work in progress. The repository contains reproducible within-session participant-identification and binary-verification experiments. Code is released under the MIT License. The dataset is not redistributed here and remains subject to the original dataset authors' terms.
This repository contains **code only**: no raw data, no derived per-participant data, no figures from the dataset's article. A summary of aggregate results is in [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md).

## Getting the data

The data are **not included** and must be obtained separately from the dataset publishers under their terms. The dataset is described in:

> Zibandehpoor, M., Alizadehziri, F., Larki, A. A., Teymouri, S. & Delrobaei, M. *Electrooculography Dataset for Objective Spatial Navigation Assessment in Healthy Participants.* Scientific Data 12, 553 (2025). doi:10.1038/s41597-025-04879-z
>
> Dataset record (as listed in the dataset's own documentation): doi:10.6084/m9.figshare.27156459 (Version 3)

Please cite both if you use the data. The dataset's documentation states it is released under CC BY 4.0; check the current terms at the source. This repository does not redistribute the dataset, and its code is not covered by the dataset's license.

### Where to put the files

Place the 27 raw recordings inside this repository folder at exactly:

```
eog-participant-identification/
└── Data/
    └── EOG Data/
        └── Raw Data/
            ├── 01.mat
            ├── 02.mat
            ├── ...
            └── 27.mat
```

- `Data/` is git-ignored, so the files stay local.
- Alternatively, keep the recordings in their existing location and set `EOG_RAW_DIR` to the `Raw Data` directory. In PowerShell: `$env:EOG_RAW_DIR = 'C:\path\to\Raw Data'`.
- Each file must contain variables `A` (horizontal EOG) and `B` (vertical EOG), sampled at 250 Hz. In the copy used for development, 26 files are MATLAB v5 and `15.mat` is MATLAB v7.3 (HDF5); both are handled.
- **File naming:** the copy used for development had files named `01.mat … 27.mat`, while the dataset's documentation shows `P_01.mat … P_27.mat`. If your copy uses the `P_XX` names, rename them, or change the filename pattern in `load_participant_signals()` in `baseline.py`.
- To check that your copy is byte-identical to the one used for the reported results, run `python inspect_eog.py`. It compares the files against [data_checksums.sha256](data_checksums.sha256). A mismatch is not necessarily an error (a different dataset version, for example), but results may differ.

If a file is missing, the scripts stop with a message that points back to this section.

## Setup

Developed and tested with **Python 3.14.7** on Windows 11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used (numpy, scipy, h5py, scikit-learn, matplotlib). Newer versions will probably work but may not reproduce the numbers exactly.

## Reproducing the experiments

All scripts resolve paths relative to their own location, so they can be run from any working directory. All random seeds are fixed.

```bash
python run_all.py                  # everything below, in order

# or one at a time (the baseline must run first; run_ablation.py checks against its output):
python baseline.py                 # baseline: chronological vs random within-session split
python run_ablation.py             # feature ablation + random-split seeds 0-99
python run_drift_analysis.py       # within-recording drift statistics
python run_coefficients.py         # logistic-regression coefficient inspection
python run_model_comparison.py     # six classifiers, fixed parameters, no tuning
python run_authentication.py       # one-vs-rest verification + unseen-impostor protocol
python test_pipeline.py            # invariant tests (or: python -m pytest test_pipeline.py)
python test_authentication.py      # data-free authentication unit tests
python inspect_eog.py              # data inspection + checksum verification
```

Outputs (CSV/JSON/PNG) are written to `results/<experiment>/`. **`results/` is git-ignored** because it contains participant-level outputs (per-window features, predictions, per-participant metrics and coefficients). Keep those local unless a privacy review approves sharing them.

The data-backed pipeline and experiment scripts require the dataset and stop with a clear "file not found" message when it is absent. `test_authentication.py` is intentionally data-free and can always exercise the verification logic on synthetic fixtures.

## Pipeline

```
raw EOG (A, B; 250 Hz, ~69 s per participant)
→ non-overlapping 2-s windows (500 EOG data points each): 34 ML samples per participant, 918 total  → 6 features per window: mean, std (ddof=0), range (max − min) of A and of B
  → per-participant split, 20 train / 7 validation / 7 test windows
       chronological holdout: first 20 / next 7 / last 7 windows (primary)
       random within-session split: windows shuffled per participant (fixed seed)
  → StandardScaler fit on training windows only
  → classifier (primary: multinomial logistic regression, C = 1)
  → accuracy, macro-F1, per-class report, 27 × 27 confusion matrix
```

Every participant appears in all three splits (closed-set identification). The validation set is reported but is not used for any selection.
## Authentication benchmark

`run_authentication.py` evaluates each participant as an authorized user against the other identities. It uses the same chronological 20/7/7 windows, fits preprocessing and class-balanced logistic regression on training data only, selects the acceptance threshold from validation scores, and evaluates once on held-out test windows. It compares all six features, means only, and std/range only. A fixed-seed random-window result is included only as a secondary temporal-optimism diagnostic.

The secondary unseen-impostor protocol uses disjoint groups of impostor identities for training, validation, and testing. This asks whether a verifier generalizes to identities absent from fitting; it is still not a cross-session experiment. See [AUTHENTICATION_PROTOCOL.md](AUTHENTICATION_PROTOCOL.md) for the exact design.

Generated outputs under `results/authentication/` include per-user and aggregate CSV/JSON metrics; frozen thresholds, confusion counts, and known-impostor test predictions; six research plots; and a generated Markdown report with actual local results. Participant-level outputs remain git-ignored pending privacy review. Without the private raw recordings, the data-free authentication tests still run, but no real experimental result is produced.
## Files

The signal-baseline follow-up is documented in [BASELINE_ROBUSTNESS_REPORT.md](BASELINE_ROBUSTNESS_REPORT.md). Run it separately from the earlier benchmarks:

```powershell
# If needed, set EOG_RAW_DIR as described above.
python run_baseline_robustness.py
python -m unittest test_baseline_robustness.py

# Preserve the first run while reproducing it in a fresh directory:
python run_baseline_robustness.py --output results/baseline_robustness_replication
```

The follow-up uses `robustness_config.json` for fixed transformations, 5,000 identity-resampling draws, and 10 matched seen/unseen-impostor partitions. It writes participant-level CSV/JSON outputs and seven plots under `results/baseline_robustness/`; existing nonempty output directories are rejected. The repository-level report contains aggregate findings and links to local figures. Resampling intervals condition on fitted models and existing recordings; they cannot establish cross-session reliability.

| File | Purpose |
|---|---|
| `baseline.py` | Core pipeline (loading, windowing, features, splits, training, evaluation) and the baseline experiment. Other scripts import it. |
| `run_ablation.py` | Feature ablation (all 6 / means only / std-range only), chronological split and random-split seeds 0–99 |
| `run_drift_analysis.py` | Per participant × feature linear-trend statistics (descriptive) |
| `run_coefficients.py` | Coefficient matrix of the six-feature chronological logistic regression |
| `run_model_comparison.py` | LR, LDA, linear SVM, RBF SVM, random forest, kNN (fixed parameters) |
| `authentication.py` | Core verification metrics, train-only modeling, thresholds, and impostor protocols. |
| `run_authentication.py` | Known- and unseen-impostor benchmark, plots, tables, and generated report. |
| `auth_config.json` | Fixed model, threshold, feature-set, seed, and unseen-impostor settings. |
| `AUTHENTICATION_PROTOCOL.md` | Verification methodology and interpretation boundary. |
| `run_all.py` | Runs all of the above, then the tests |
| `test_pipeline.py` | 15 data-backed invariant tests: 27 IDs; raw files unmodified; 34 windows each; 918 × 6 finite features; no window overlap; disjoint splits with all IDs; train-only scaling; chronological order; seed determinism; ablation columns; coefficient-row mapping; baseline accuracy |
| `test_authentication.py` | Data-free synthetic tests for metrics, scaling, threshold isolation, identity splits, and reproducibility. |
| `inspect_eog.py` | Prints per-file format, length and window count; verifies checksums |
| `data_checksums.sha256` | SHA-256 of the 27 raw files used for the reported results. Hashes only, no data. |
| `RESEARCH_SUMMARY.md` | Aggregate results and limitations |
| `RELEASE_READINESS.md` | Release checklist and pending decisions |

## Scientific limitations

- **One session per participant.** Within-session identification only. Nothing here demonstrates identity that is stable across sessions.
- **Recording-level DC offsets.** Channel means differ greatly between recordings, and the classifier relies heavily on them. Their origin (electrode, amplifier or session factors versus physiology) cannot be determined from this dataset.
- **Within-recording drift.** Mean features drift slowly and nearly monotonically within most recordings.
- **Random within-session splits are more optimistic.** They consistently produced higher accuracy than chronological holdout. Because test windows are interleaved in time with training windows, slowly drifting recording-specific characteristics may contribute to this difference.- **Repeated random splits are not independent replications.** They are repartitions of the same 27 recordings.
- **Windows are not independent.** All 34 windows of a participant come from one ~69-s recording.
- **Small sample.** 27 participants and 7 test windows each; one test window ≈ 0.5 percentage points of accuracy.
- **Coefficient magnitudes are descriptive.** They show how strongly the fitted linear model uses a standardized feature, not causal or physiological importance.

## License

The analysis code in this repository is released under the MIT License. See [LICENSE](LICENSE) for details.

The EOG dataset is a separate work and is **not** covered by this repository's MIT License. The dataset is not redistributed here. Users should obtain it from the original source and follow the dataset authors' license and citation requirements.

## Acknowledgments

This project is conducted under the mentorship of **Professor Qingqing Li at Towson University**.

I thank Professor Li for guidance on experimental design, machine-learning methodology, interpretation of the results, and the development of the biometric-authentication direction of this project.

I also acknowledge the authors of the EOG dataset used in this project, whose publicly available data enable these experiments.
