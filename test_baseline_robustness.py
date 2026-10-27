"""Data-free tests for signal controls, matched panels, and identity uncertainty."""

import unittest

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import authentication as auth
import baseline as lb
import baseline_robustness as br


class RobustnessTests(unittest.TestCase):
    def test_constant_additive_shift_invariance(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(2, 500))
        for method in ['window_centered', 'first_sample_reference', 'causal_dc_blocker']:
            original = br.transformed_features(x, y, method)
            shifted = br.transformed_features(x + 127, y - 91, method)
            np.testing.assert_allclose(original, shifted, atol=1e-12)

    def test_centering_preserves_spread_and_zeroes_means_exactly(self):
        x = np.linspace(-2, 7, 500) + 10000
        y = np.sin(np.linspace(0, 6, 500)) - 500
        raw = np.asarray(lb.window_features(x, y))
        centered = br.transformed_features(x, y, 'window_centered')
        np.testing.assert_array_equal(centered[[0, 3]], [0., 0.])
        np.testing.assert_allclose(centered[[1, 2, 4, 5]], raw[[1, 2, 4, 5]])

    def test_causal_transforms_do_not_use_future_samples(self):
        x = np.linspace(1, 4, 500)
        altered = x.copy()
        altered[100:] += 100
        for method in ['first_sample_reference', 'causal_dc_blocker']:
            before = br.transform_window(x, method)
            after = br.transform_window(altered, method)
            np.testing.assert_array_equal(before[:100], after[:100])

    def test_dc_blocker_zero_on_constant_input_and_trial_reset(self):
        np.testing.assert_array_equal(br.transform_window(np.full(500, 123.), 'causal_dc_blocker'),
                                      np.zeros(500))
        x = np.arange(500, dtype=float)
        self.assertEqual(br.transform_window(x, 'causal_dc_blocker')[0], 0.)
        np.testing.assert_array_equal(br.transform_window(x, 'causal_dc_blocker'),
                                      br.transform_window(x, 'causal_dc_blocker'))

    def test_bootstrap_draws_complete_identities_and_pairs_exactly(self):
        a = br.bootstrap_counts(27, 200, 31)
        b = br.bootstrap_counts(27, 200, 31)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a.sum(axis=1), np.full(200, 27))
        records = [{'participant': i + 1, 'roc_auc': .5 + i / 100} for i in range(27)]
        values = br.cluster_bootstrap(records, a, 'roc_auc')
        np.testing.assert_array_equal(values - br.cluster_bootstrap(records, b, 'roc_auc'), np.zeros(200))
        self.assertAlmostEqual(values[0], sum(a[0, i] * records[i]['roc_auc'] for i in range(27)) / 27)

    def test_identity_weighted_auc_and_eer_match_independent_sklearn_calculation(self):
        # Tied scores plus unequal identity weights exercise grouped ROC points.
        y = np.array([1, 1, 2, 2, 3, 3])
        scores = np.array([.9, .4, .4, .1, .8, .2])
        truth = (y == 1).astype(int)
        detail = {'test_scores': scores, 'y_test': truth, 'test_indices': np.arange(6), 'threshold': .4}
        counts = np.array([[1, 1, 1], [1, 2, 0], [2, 0, 1]])
        metrics = br.identity_weighted_metrics(detail, y, 1, [1, 2, 3], counts)
        for i, weights in enumerate(counts):
            observation_weights = weights[y - 1].astype(float)
            self.assertAlmostEqual(metrics['roc_auc'][i], roc_auc_score(truth, scores, sample_weight=observation_weights))
            fpr, tpr, _ = roc_curve(truth, scores, sample_weight=observation_weights, drop_intermediate=False)
            fnr = 1 - tpr
            high = np.where(fpr >= fnr)[0][0]
            low = high - 1
            fraction = (fnr[low] - fpr[low]) / ((fpr[high] - fnr[high]) - (fpr[low] - fnr[low]))
            independent_eer = fpr[low] + fraction * (fpr[high] - fpr[low])
            self.assertAlmostEqual(metrics['eer'][i], independent_eer)
        self.assertAlmostEqual(metrics['eer'][0], auth.equal_error_rate(truth, scores)[0])

    def test_matched_panels_equal_counts_and_exact_validation_test_rows(self):
        y = np.repeat(np.arange(1, 28), 34)
        windows = np.tile(np.arange(34), 27)
        split = lb.make_split(y, windows, 'chronological')
        for target in [1, 13, 27]:
            groups = auth.make_unseen_identity_split(target, list(range(1, 28)), 77)
            unseen = br.matched_impostor_indices(y, split, target, groups, False)
            seen = br.matched_impostor_indices(y, split, target, groups, True)
            self.assertEqual([len(unseen[k]) for k in ['train', 'val', 'test']], [340, 42, 42])
            self.assertEqual([len(seen[k]) for k in ['train', 'val', 'test']], [340, 42, 42])
            np.testing.assert_array_equal(unseen['val'], seen['val'])
            np.testing.assert_array_equal(unseen['test'], seen['test'])
            test_ids = set(y[unseen['test']]) - {target}
            self.assertFalse(test_ids & set(y[unseen['train']]))
            self.assertTrue(test_ids <= set(y[seen['train']]))
            for panel in [seen, unseen]:
                self.assertFalse(set(panel['train']) & set(panel['test']))
                self.assertFalse(set(panel['train']) & set(panel['val']))

    def test_threshold_deletion_selection_does_not_depend_on_test_scores(self):
        y = np.array([1, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3])
        detail = {'validation_indices': np.arange(7), 'y_validation': (y[:7] == 1).astype(int),
                  'validation_scores': np.array([.9, .7, .4, .6, .1, .3, .2]),
                  'y_test': (y[7:] == 1).astype(int), 'test_scores': np.linspace(.1, .9, 6)}
        rows = br.threshold_deletions(detail, y)
        altered = {**detail, 'test_scores': detail['test_scores'] * -100}
        changed = br.threshold_deletions(altered, y)
        self.assertEqual(len(rows), 5)
        self.assertEqual([r['threshold'] for r in rows], [r['threshold'] for r in changed])


if __name__ == '__main__':
    unittest.main()
