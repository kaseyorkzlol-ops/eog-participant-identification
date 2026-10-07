# Release Readiness

**Current state:** private GitHub repository, code only. **Not approved for public release.** Do not make this repository public, add a license, or publish a release until the pending items below are resolved.

## Checklist

| Item | Status | Notes |
|---|---|---|
| Code-only repository (no dataset files) | ✅ Done | No `.mat` files, `Data/` directory, demographic tables or article figures. `Data/`, `*.mat`, `*.h5` and `*.csv` are git-ignored. |
| Participant-level outputs excluded | ✅ Done | `results/` (per-window features, predictions, per-participant metrics, coefficients, drift fits) is git-ignored and was never committed. The docs use aggregate numbers only. |
| No fitted models or derived data committed | ✅ Done | The code does not save models. `*.pkl`, `*.joblib`, `*.npy` and `*.npz` are ignored as a safeguard. |
| No credentials, environment files, personal paths or caches | ✅ Done | The staged files were scanned for keys, tokens, passwords, emails, usernames and absolute paths before the initial commit. None were found. |
| Dependencies documented | ✅ Done | `requirements.txt` pins the exact tested versions. Tested with Python 3.14.7 on Windows 11 only. |
| Setup and reproduction instructions | ✅ Done | `README.md`, including where to place the separately obtained data. |
| Code compiles without data | ✅ Done | All `.py` files compile. Without the dataset, the scripts stop with a clear "file not found" message. |
| Tests pass **with data supplied separately** | ✅ Done, locally only | An export of exactly the committed files, with the raw data added outside version control, ran `run_all.py` in a fresh venv built from `requirements.txt`. All 15 tests passed. 36 of 37 result files were byte-identical to the original project; the 37th differs only in a module-name string. |
| Tests in this repository alone (no data) | ⚠️ Cannot run | Every test needs the raw dataset. **No CI is configured**, and the tests cannot run on GitHub without the data. |
| Tested on macOS / Linux / other Python versions | ❌ Not done | |
| Copied third-party code | ✅ None | All code was written for this project. The companion MATLAB analysis repository referenced in the dataset documentation was **not** copied. |
| Dataset citation preserved | ✅ Done | The data descriptor and the dataset record DOI are in `README.md`, taken from the dataset's own documentation. |
| Code license | ⛔ PENDING | No license has been chosen. None is included; all rights are reserved by default. |
| Authorship / attribution | ⛔ PENDING | No author list, copyright line or `CITATION.cff` for this analysis has been added. Commit metadata uses the repository owner's git identity with a GitHub no-reply address. The initial commit carries an AI-assistance co-author trailer. |
| Mention or acknowledgement of the supervising professor | ⛔ PENDING | The supervising professor is deliberately **not named** anywhere in this repository (including file names) until they confirm whether and how they wish to be mentioned. |
| Dataset permissions | ⛔ PENDING | The dataset's documentation states CC BY 4.0. This was not verified against the live dataset record. Confirm that this use, and any future sharing of derived outputs, complies with it. |
| Privacy review | ⛔ PENDING | Needed before sharing anything participant-level: results, per-participant metrics, or the demographic data that ships with the dataset. `data_checksums.sha256` contains file hashes and relative file names only. `run_coefficients.py` and `run_drift_analysis.py` hard-code a few pseudonymous participant IDs, which choose what to plot and inspect; their comments were made neutral. |
| Scientific limitations documented | ✅ Done | `README.md` and `RESEARCH_SUMMARY.md`: within-session only, one recording per participant, DC offsets, drift, optimistic random splits, non-independent repeated splits, descriptive coefficients. |

## Decisions needed before any public release

1. Choose a code license, or decide to keep the code unlicensed and private.
2. Confirm authorship and attribution for the analysis, and whether to add a `CITATION.cff` for it.
3. Confirm whether and how the supervising professor is named or acknowledged.
4. Confirm the dataset terms and whether any derived outputs may be shared.
5. Complete a privacy review of participant-level outputs.
6. Decide whether to add data-free unit tests and CI, for example tests on synthetic signals.
