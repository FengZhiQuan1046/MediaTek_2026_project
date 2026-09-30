import unittest
from types import SimpleNamespace

import numpy as np

from experiment import (bootstrap_mean, fixed_cohort, historical_summary,
                        history_uniform_cluster_chance,
                        paired_decay, pairwise_accuracy, sample_matches,
                        subset_cosine_normalizer, summarize_information_coverage,
                        summarize_lags)


class InverseHistoryTests(unittest.TestCase):
    def test_training_prefix_and_held_out_target_keep_their_roles(self):
        from src.data import InteractionData
        from src.train_mamba_rl import Transition, history_batch, evaluation_history_batch
        data = InteractionData({0: [1, 2, 3, 4]}, {0: 5}, {0: 6},
                               [str(i) for i in range(7)], 1, 7)
        data.reverse_history_input = True
        batch, lengths, targets = history_batch(data, [Transition(0, 3, 4)], 2, "cpu")
        self.assertEqual(batch[0, :lengths[0]].tolist(), [3, 2])
        self.assertEqual(targets.tolist(), [4])
        valid, _, raw_valid = evaluation_history_batch(data, [0], "valid", 3, "cpu")
        test, _, raw_test = evaluation_history_batch(data, [0], "test", 3, "cpu")
        self.assertEqual(valid[0].tolist(), [4, 3, 2])
        self.assertEqual(test[0].tolist(), [5, 4, 3])
        self.assertEqual(raw_valid, [[2, 3, 4]])
        self.assertEqual(raw_test, [[3, 4, 5]])
        self.assertEqual(data.train_by_user[0], [1, 2, 3, 4])

    def test_deletion_lag_follows_reversed_model_input(self):
        import csv
        import tempfile
        import torch
        from pathlib import Path
        from src.data import InteractionData
        from src.model_mamba_rl import MultiAgentMambaRecommender
        from src.train_mamba_rl import measure_history_item_influence
        data = InteractionData({0: [1, 2, 3]}, {0: 4}, {0: 5},
                               [str(i) for i in range(9)], 1, 9)
        data.reverse_history_input = True
        model = MultiAgentMambaRecommender(torch.randn(9, 8), dim=8,
                                           use_graph_embeddings=False,
                                           use_short=False, use_preference=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "influence.csv"
            count = measure_history_item_influence(
                model, data, "test", 2, 4, "cpu", path,
                negative_count=2, user_limit=1, seed=1, dataset="synthetic"
            )
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(count, 4)
        self.assertEqual([(int(r["lag"]), int(r["item"])) for r in rows],
                         [(1, 1), (2, 2), (3, 3), (4, 4)])
        self.assertEqual(data.train_by_user[0], [1, 2, 3])

    def test_original_order_is_unchanged_without_inverse_flag(self):
        from src.data import InteractionData
        from src.train_mamba_rl import Transition, history_batch
        data = InteractionData({0: [1, 2, 3]}, {0: 4}, {0: 5},
                               [str(i) for i in range(6)], 1, 6)
        batch, _, target = history_batch(data, [Transition(0, 3, 4)], 2, "cpu")
        self.assertEqual(batch[0].tolist(), [2, 3])
        self.assertEqual(target.tolist(), [4])


class StatisticsTests(unittest.TestCase):
    def test_lag_one_and_fixed_cohort(self):
        rows = [
            {"user": 1, "lag": 1, "history_length": 3, "excess_cosine": .2},
            {"user": 1, "lag": 3, "history_length": 3, "excess_cosine": -.1},
            {"user": 2, "lag": 1, "history_length": 1, "excess_cosine": .5},
        ]
        self.assertEqual(fixed_cohort(rows, 3), [rows[0], rows[1]])
        self.assertAlmostEqual(historical_summary(rows, 1), .35)

    def test_history_uniform_cluster_chance(self):
        groups = np.array([2, 5, 9])
        self.assertAlmostEqual(history_uniform_cluster_chance(groups, 5), 1 / 3)
        self.assertEqual(history_uniform_cluster_chance(groups, 7), 0.0)
        self.assertEqual(history_uniform_cluster_chance(np.array([2]), 2), 1.0)

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
            "raw_cosine_z", "random_cosine_z", "excess_cosine_z",
            "random_cluster_match", "excess_cluster_match",
            "history_normalized_cluster_match")}}
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
                             "excess_cosine_z": 1.0 if lag == 1 else 0.0,
                             "excess_cluster_match": 0.0})
        result = paired_decay(rows, SimpleNamespace(bootstrap=10, seed=1), "toy", "test")
        self.assertEqual(result[0]["n_users"], 1)
        self.assertEqual(result[0]["mean"], 1.0)

    def test_subset_normalizer_uses_distinct_training_pairs(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0],
                            [2 ** -.5, 2 ** -.5]], dtype=np.float32)
        mean, std, count = subset_cosine_normalizer(
            vectors, 6, np.random.default_rng(1))
        self.assertGreaterEqual(mean, 0.0)
        self.assertGreater(std, 0.0)
        self.assertEqual(count, 6)

    def test_information_coverage_uses_positive_signal(self):
        rows = [
            {"user": 1, "lag": 1, "history_length": 3, "excess_cosine_z": 3.0},
            {"user": 1, "lag": 2, "history_length": 3, "excess_cosine_z": -4.0},
            {"user": 1, "lag": 3, "history_length": 3, "excess_cosine_z": 1.0},
            {"user": 2, "lag": 1, "history_length": 1, "excess_cosine_z": -1.0},
        ]
        summary, users = summarize_information_coverage(rows, "toy", "test", 3)
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["items_for_50pct"], 1)
        self.assertEqual(users[0]["items_for_80pct"], 3)
        self.assertEqual(summary[0]["mean_percent"], 75.0)
        self.assertEqual(summary[2]["mean_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
