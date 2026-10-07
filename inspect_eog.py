"""
Quick inspection of the raw EOG files (no model training).

For each participant prints the MATLAB format, the number of samples, the
number of 2-s windows, and the first window's six features, then checks the
raw files against data_checksums.sha256.

This file originally contained the first draft of the loading / feature /
split code. Its main block referenced X_train_list etc. without defining them
and raised NameError. The working parts (loading v5 and v7.3 .mat files,
2-s windows, six features) now live in baseline.py; this script reuses
them and is kept only as a lightweight data check.

Run:  python inspect_eog.py
"""

import hashlib
import os

import numpy as np

import baseline as lb

CHECKSUM_FILE = os.path.join(lb.REPO_DIR, "data_checksums.sha256")


def verify_checksums():
    """Return list of (relative path, ok) for every file listed in data_checksums.sha256."""
    results = []
    with open(CHECKSUM_FILE) as fh:
        for line in fh:
            if not line.strip():
                continue
            expected, rel = line.strip().split("  ", 1)
            path = os.path.join(lb.REPO_DIR, rel)
            if not os.path.exists(path):
                results.append((rel, False))
                continue
            with open(path, "rb") as f:
                results.append((rel, hashlib.sha256(f.read()).hexdigest() == expected))
    return results


def main():
    print(f"Raw data folder: {os.path.relpath(lb.RAW_DIR, lb.REPO_DIR)}")
    print(f"{'ID':<5}{'format':<24}{'samples':>8}{'seconds':>9}{'windows':>9}   first-window features")
    for pid in lb.PARTICIPANT_IDS:
        A, B, fmt = lb.load_participant_signals(pid)
        A_win, B_win, _ = lb.extract_windows(A, B)
        f0 = np.round(lb.window_features(A_win[0], B_win[0]), 3).tolist()
        print(f"P{pid:02d}  {fmt:<24}{len(A):>8}{len(A) / lb.FS:>9.2f}{len(A_win):>9}   {f0}")

    checks = verify_checksums()
    bad = [rel for rel, ok in checks if not ok]
    print(f"\nChecksums: {len(checks) - len(bad)}/{len(checks)} raw files match data_checksums.sha256")
    for rel in bad:
        print("  MISMATCH or missing:", rel)


if __name__ == "__main__":
    main()
