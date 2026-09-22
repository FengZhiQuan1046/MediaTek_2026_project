import unittest
from types import SimpleNamespace

import numpy as np

from experiment import bootstrap_mean, fixed_cohort, historical_summary, paired_decay, pairwise_accuracy, sample_matches, summarize_lags


class StatisticsTests(unittest.TestCase):
    def test_lag_one_and_fixed_cohort(self):
        rows = [
            {"user": 1, "lag": 1, "history_length": 3, "excess_cosine": .2},
            {"user": 1, "lag": 3, "history_length": 3, "excess_cosine": -.1},
            {"user": 2, "lag": 1, "history_length": 1, "excess_cosine": .5},
        ]
        self.assertEqual(fixed_cohort(rows, 3), [rows[0], rows[1]])
        self.assertAlmostEqual(historical_summary(rows, 1), .35)

    def test_pairwise_ties(self):
        self.assertEqual(pairwise_accuracy(1., np.array([0., 1., 2.])), .5)

    def test_user_bootstrap_is_deterministic(self):
        values = np.array([0., 1., 2., 3.])
        a = bootstrap_mean(values, 200, 4)
        b = bootstrap_mean(values, 200, 4)
        self.assertEqual(a, b)
        self.assertEqual(a[0], 1.5)

    def test_fixed_cohort_never_includes_lags_beyond_k(self):
        row = {"user": 1, "history_length": 32, **{name: 0.0 for name in (
            "raw_cosine", "random_cosine", "excess_cosine", "raw_cluster_match",
            "random_cluster_match", "excess_cluster_match")}}
        rows = [{**row, "lag": lag} for lag in range(1, 33)]
        summary = summarize_lags(rows, SimpleNamespace(bootstrap=5, seed=1), "toy", "test")
        for record in summary:
            if record["cohort"].startswith("fixed_K"):
                self.assertLessEqual(record["lag_end"], int(record["cohort"][7:]))

    def test_reference_matching_excludes_history_not_future_target(self):
        bins = np.array([0, 0, 0, 1], dtype=np.int8)
        pools = [np.array([0, 1, 2]), np.array([3]), np.array([], dtype=int),
                 np.array([], dtype=int), np.array([], dtype=int)]
        draws, widening = sample_matches(0, np.array([0, 1]), bins, pools,
                                          20, np.random.default_rng(1))
        self.assertEqual(widening, 0)
        self.assertTrue(np.all(draws == 2))  # item 2 may be the held-out target.
        negatives, _ = sample_matches(0, np.array([0, 1]), bins, pools,
                                      20, np.random.default_rng(1), exclude_target=2)
        self.assertTrue(np.all(negatives == 3))

    def test_paired_decay_uses_identical_users(self):
        rows = []
        for user, length in ((1, 16), (2, 8)):
            for lag in range(1, length + 1):
                rows.append({"user": user, "history_length": length, "lag": lag,
                             "excess_cosine": 1.0 if lag == 1 else 0.0,
                             "excess_cluster_match": 0.0})
        result = paired_decay(rows, SimpleNamespace(bootstrap=10, seed=1), "toy", "test")
        self.assertEqual(result[0]["n_users"], 1)
        self.assertEqual(result[0]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
