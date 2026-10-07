"""
Invariant checks for the EOG participant-identification pipeline.

Run (either works; pytest is optional and not a project dependency):
    python test_pipeline.py
    python -m pytest test_pipeline.py
"""

import hashlib
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

import baseline as lb

X, y, window_index, start_sample, info = lb.build_dataset()


# ---------------------------------------------------------------- raw data
def test_27_identities_discovered():
    assert sorted(np.unique(y).tolist()) == list(range(1, 28))


def test_expected_raw_files_present():
    names = sorted(f for f in os.listdir(lb.RAW_DIR) if f.endswith(".mat"))
    assert names == [f"{p:02d}.mat" for p in range(1, 28)]


def test_raw_data_files_not_modified():
    # data_checksums.sha256 records the SHA-256 of the raw files used for the reported results
    with open(os.path.join(lb.REPO_DIR, "data_checksums.sha256")) as fh:
        entries = [line.strip().split("  ", 1) for line in fh if line.strip()]
    assert len(entries) == 27
    for expected, rel in entries:
        with open(os.path.join(lb.REPO_DIR, rel), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == expected, f"{rel} differs from recorded checksum"


# ---------------------------------------------------------------- windows / features
def test_34_windows_per_participant():
    assert all(r["n_windows"] == 34 for r in info)
    assert np.all(np.bincount(y)[1:] == 34)


def test_feature_matrix_shape_and_finite():
    assert X.shape == (918, 6) and y.shape == (918,)
    assert X.shape[1] == len(lb.FEATURE_NAMES)
    assert np.isfinite(X).all()


def test_windows_do_not_overlap():
    for pid in lb.PARTICIPANT_IDS:
        starts = np.sort(start_sample[y == pid])
        assert np.all(np.diff(starts) == lb.WINDOW_SIZE)   # back-to-back, no overlap
        assert starts[0] == 0


def test_features_match_raw_signal():
    # recompute one window directly from the raw file
    A, B, _ = lb.load_participant_signals(15)          # the v7.3 / h5py file
    row = np.where((y == 15) & (window_index == 3))[0][0]
    a, b = A[1500:2000], B[1500:2000]
    expected = [a.mean(), a.std(), a.max() - a.min(), b.mean(), b.std(), b.max() - b.min()]
    assert np.allclose(X[row], expected)


def test_feature_ablation_column_selection():
    for name, cols in lb.FEATURE_SETS.items():
        Xs = lb.select_features(X, cols)
        assert Xs.shape == (918, len(cols))
        for k, c in enumerate(cols):
            assert np.array_equal(Xs[:, k], X[:, lb.FEATURE_NAMES.index(c)])
    assert lb.FEATURE_SETS["means_only"] == ["A_mean", "B_mean"]
    assert lb.FEATURE_SETS["std_range_only"] == ["A_std", "A_range", "B_std", "B_range"]
    assert lb.FEATURE_SETS["all_6"] == lb.FEATURE_NAMES


# ---------------------------------------------------------------- splits
def test_splits_disjoint_complete_and_all_identities_present():
    for mode in ("chronological", "random"):
        split = lb.make_split(y, window_index, mode)
        lb.check_split(split, y, len(y))   # raises on overlap, missing rows, or missing identities
        for k in ("train", "val", "test"):
            assert set(y[split[k]].tolist()) == set(lb.PARTICIPANT_IDS)


def test_split_sizes_20_7_7():
    for mode in ("chronological", "random"):
        split = lb.make_split(y, window_index, mode)
        for k, n in (("train", 20), ("val", 7), ("test", 7)):
            assert np.all(np.bincount(y[split[k]])[1:] == n)


def test_chronological_split_preserves_time_order():
    split = lb.make_split(y, window_index, "chronological")
    for pid in lb.PARTICIPANT_IDS:
        tr = window_index[split["train"]][y[split["train"]] == pid]
        va = window_index[split["val"]][y[split["val"]] == pid]
        te = window_index[split["test"]][y[split["test"]] == pid]
        assert tr.max() < va.min() and va.max() < te.min()
        assert np.array_equal(np.sort(tr), np.arange(20))


def test_random_split_is_reproducible_and_seed_dependent():
    s1 = lb.make_split(y, window_index, "random", seed=lb.SEED)
    s2 = lb.make_split(y, window_index, "random", seed=lb.SEED)
    s3 = lb.make_split(y, window_index, "random", seed=lb.SEED + 1)
    for k in ("train", "val", "test"):
        assert np.array_equal(s1[k], s2[k])
    assert not np.array_equal(np.sort(s1["test"]), np.sort(s3["test"]))


# ---------------------------------------------------------------- preprocessing / model
def test_scaler_fit_on_training_data_only():
    split = lb.make_split(y, window_index, "chronological")
    scaler, _ = lb.train_model(X[split["train"]], y[split["train"]])
    ref = StandardScaler().fit(X[split["train"]])
    assert scaler.n_samples_seen_ == 540
    assert np.allclose(scaler.mean_, ref.mean_) and np.allclose(scaler.scale_, ref.scale_)
    # and it must differ from a scaler fit on everything (which would be leakage)
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_coefficient_rows_map_to_participants():
    import run_coefficients
    scaler, model, Xc, yc, split = run_coefficients.fit_chronological_baseline()
    assert model.classes_.tolist() == lb.PARTICIPANT_IDS
    assert model.coef_.shape == (27, 6)
    # using row i as "class model.classes_[i]" must reproduce model.predict exactly
    Z = scaler.transform(Xc[split["test"]])
    manual = model.classes_[np.argmax(Z @ model.coef_.T + model.intercept_, axis=1)]
    assert np.array_equal(manual, model.predict(Z))


def test_baseline_reproduces_reported_accuracy():
    split = lb.make_split(y, window_index, "chronological")
    m = lb.run_experiment(X, y, split)["metrics"]
    assert round(m["test_accuracy"], 4) == 0.5450
    assert round(m["val_accuracy"], 4) == 0.5926


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"All {len(tests)} checks passed.")
