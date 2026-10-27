"""Data-free invariant tests for the authentication benchmark."""

import numpy as np

import authentication as auth


def fixture():
    rng = np.random.default_rng(41)
    participants = list(range(1, 7))
    rows, labels, windows = [], [], []
    for pid in participants:
        for window in range(34):
            rows.append([pid + rng.normal(0, .25), pid * .2 + rng.normal(0, .3),
                         np.sin(window / 5) + rng.normal(0, .2)])
            labels.append(pid)
            windows.append(window)
    X, y, windows = np.asarray(rows), np.asarray(labels), np.asarray(windows)
    split = {}
    for name, bounds in {"train": (0, 20), "val": (20, 27), "test": (27, 34)}.items():
        split[name] = np.where((windows >= bounds[0]) & (windows < bounds[1]))[0]
    return X, y, windows, split, participants


def test_equal_error_rate_and_threshold_metrics():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    eer, threshold = auth.equal_error_rate(y, scores)
    metrics, predicted = auth.binary_metrics(y, scores, threshold)
    assert eer == 0.0
    assert metrics["far"] == 0.0 and metrics["frr"] == 0.0
    assert metrics["tn"] == 2 and metrics["tp"] == 2
    assert np.array_equal(predicted, y)


def test_known_protocol_counts_and_train_only_scaling():
    X, y, _, split, participants = fixture()
    records, details = auth.run_known_impostor(
        X, y, split, participants, 7, {"C": 1, "class_weight": "balanced", "max_iter": 1000})
    assert len(records) == 6
    assert all(r["n_genuine"] == 7 and r["n_impostor"] == 35 for r in records)
    assert all(np.isfinite(r["threshold"]) for r in records)
    assert np.allclose(details[1]["scaler_mean"], X[split["train"]].mean(axis=0))
    assert not np.allclose(details[1]["scaler_mean"], X.mean(axis=0))


def test_test_values_cannot_change_selected_threshold():
    X, y, _, split, participants = fixture()
    config = {"C": 1, "class_weight": "balanced", "max_iter": 1000}
    first, _ = auth.run_known_impostor(X, y, split, participants, 9, config)
    altered = X.copy()
    altered[split["test"]] += 1000
    second, _ = auth.run_known_impostor(altered, y, split, participants, 9, config)
    assert np.allclose([r["threshold"] for r in first],
                       [r["threshold"] for r in second])


def test_unseen_identity_partitions_are_disjoint_complete_and_deterministic():
    _, _, _, _, participants = fixture()
    first = auth.make_unseen_identity_split(1, participants, 77, counts=(3, 1, 1))
    second = auth.make_unseen_identity_split(1, participants, 77, counts=(3, 1, 1))
    sets = [set(first[k].tolist()) for k in ("train", "val", "test")]
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(participants) - {1}


def test_unseen_observations_have_no_impostor_identity_overlap():
    _, y, _, split, participants = fixture()
    groups = auth.make_unseen_identity_split(2, participants, 8, counts=(3, 1, 1))
    indices = auth.unseen_observation_indices(y, split, 2, groups)
    observed = {k: set(y[indices[k]].tolist()) - {2} for k in indices}
    assert observed == {k: set(v.tolist()) for k, v in groups.items()}
    assert not (observed["train"] & observed["val"] or
                observed["train"] & observed["test"] or
                observed["val"] & observed["test"])
    assert sum(y[indices["train"]] == 2) == 20
    assert sum(y[indices["val"]] == 2) == 7
    assert sum(y[indices["test"]] == 2) == 7


def test_reproducible_known_protocol():
    X, y, _, split, participants = fixture()
    config = {"C": 1, "class_weight": "balanced", "max_iter": 1000}
    a, _ = auth.run_known_impostor(X, y, split, participants, 123, config)
    b, _ = auth.run_known_impostor(X, y, split, participants, 123, config)
    for ra, rb in zip(a, b):
        for key in ("threshold", *auth.METRIC_NAMES):
            assert ra[key] == rb[key]


def test_complete_unseen_protocol_counts_and_reproducibility():
    X, y, _, split, participants = fixture()
    config = {"C": 1, "class_weight": "balanced", "max_iter": 1000}
    a, _, assignments_a = auth.run_unseen_impostor(
        X, y, split, participants, 51, 2, (3, 1, 1), config)
    b, _, assignments_b = auth.run_unseen_impostor(
        X, y, split, participants, 51, 2, (3, 1, 1), config)
    assert len(a) == 12 and len(assignments_a) == 60
    assert all(r["n_genuine"] == 7 and r["n_impostor"] == 34 for r in a)
    assert assignments_a == assignments_b
    assert all(ra["threshold"] == rb["threshold"] and ra["roc_auc"] == rb["roc_auc"]
               for ra, rb in zip(a, b))


if __name__ == "__main__":
    tests = [value for name, value in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"All {len(tests)} authentication checks passed.")
