import unittest
from types import SimpleNamespace
from src.sequence_length_metrics import SequenceLengthMetrics


class SequenceLengthMetricsTest(unittest.TestCase):
    def test_boundary_and_empty_group(self):
        stats = SequenceLengthMetrics({0: [1] * 10, 1: [1] * 11}, {0: 2, 1: 2}, 10)
        stats.update([0], [1])
        report = stats.report()
        self.assertEqual(report['short']['recall@10'], 1)
        self.assertEqual(report['long']['total_users'], 1)
        self.assertIsNone(report['long']['recall@10'])
        empty = SequenceLengthMetrics({0: [1]}, {0: 2}, 10).report()
        self.assertEqual(empty['long']['total_users'], 0)
        self.assertIsNone(empty['long']['history_length_mean'])

    def test_full_sequence_boundary_and_periodic_files(self):
        import json
        import tempfile
        from pathlib import Path
        import torch
        from src.model_mamba_rl import MultiAgentMambaRecommender
        from src.train_mamba_rl import evaluate
        from src.sequence_length_metrics import PeriodicTestScores
        model = MultiAgentMambaRecommender(torch.randn(20, 8), dim=8, use_graph_embeddings=False)
        data = SimpleNamespace(
            train_by_user={0: list(range(8)), 1: list(range(9))},
            valid_target={0: 10, 1: 11}, test_target={0: 12, 1: 13}, num_items=20,
            evaluation_length_threshold=10, evaluation_length_basis='filtered_full_sequence',
        )
        metrics, _ = evaluate(model, data, 'test', 2, 2, 'cpu')
        groups = metrics['sequence_length']
        self.assertEqual(groups['short']['history_length_max'], 10)
        self.assertEqual(groups['long']['history_length_min'], 11)
        self.assertEqual(groups['short']['evaluated_users'], 1)
        self.assertEqual(groups['long']['evaluated_users'], 1)
        with tempfile.TemporaryDirectory() as directory:
            writer = PeriodicTestScores(directory, {'dataset': 'synthetic'}, 'test')
            writer.record(metrics, 'specialists', 1, 10)
            writer.record(metrics, 'joint', 1, 20)
            for group in ('short', 'long'):
                payload = json.loads((Path(directory) / f'{group}_scores.json').read_text())
                self.assertEqual(len(payload['test']), 6)
                self.assertEqual([x['global_step'] for x in payload['periodic_test_history']], [10, 20])
                self.assertEqual(payload['length_basis'], 'filtered_full_sequence')

    def test_evaluation_preserves_overall_and_weighted_metrics(self):
        import torch
        from src.model_mamba_rl import MultiAgentMambaRecommender
        from src.train_mamba_rl import evaluate
        torch.manual_seed(5)
        model = MultiAgentMambaRecommender(torch.randn(20, 8), dim=8, lora_rank=2, short_window=2, use_graph_embeddings=False)
        data = SimpleNamespace(
            train_by_user={0: [1, 2], 1: [2, 3, 4], 2: [3, 4, 5, 6]},
            valid_target={0: 7, 1: 8, 2: 9}, test_target={0: 10, 1: 11, 2: 12}, num_items=20,
        )
        for split in ('valid', 'test'):
            if hasattr(data, 'evaluation_length_threshold'):
                del data.evaluation_length_threshold
            baseline, _ = evaluate(model, data, split, 2, 2, 'cpu')
            self.assertNotIn('sequence_length', baseline)
            data.evaluation_length_threshold = 2
            grouped, _ = evaluate(model, data, split, 2, 2, 'cpu')
            groups = grouped['sequence_length']
            self.assertEqual(groups['short']['total_users'], 1)
            self.assertEqual(groups['long']['total_users'], 2)
            for metric in ('recall@5', 'recall@10', 'ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'):
                self.assertEqual(baseline[metric], grouped[metric])
                weighted = sum(groups[g][metric] * groups[g]['evaluated_users'] for g in ('short', 'long')) / 3
                self.assertAlmostEqual(weighted, grouped[metric], places=6)


if __name__ == '__main__':
    unittest.main()
