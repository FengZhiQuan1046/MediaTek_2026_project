from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MEDIATEK = ROOT.parents[1]
VER4 = MEDIATEK / "ver4"
for path in (ROOT, VER4):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from data_adapter import evaluation_batch, interaction_cache_path, ranking_metrics  # noqa: E402
from model import Mamba4Rec  # noqa: E402
from src.data import build_data, synthetic_events  # noqa: E402
from src.train_mamba_rl import evaluation_history_batch  # noqa: E402


class IdentityMixer(nn.Module):
    def __init__(self, **kwargs): super().__init__()
    def forward(self, inputs): return inputs


class Mamba4RecTest(unittest.TestCase):
    def setUp(self): self.data = build_data(synthetic_events())

    def test_evaluation_histories_match_ver4(self):
        users = sorted(self.data.test_target)[:5]
        for split in ("valid", "test"):
            sequences, lengths, histories = evaluation_batch(self.data, users, split, 2, "cpu")
            expected_sequences, expected_lengths, expected = evaluation_history_batch(self.data, users, split, 2, "cpu")
            self.assertEqual(histories, expected)
            for row, history in enumerate(expected):
                self.assertEqual((sequences[row, :lengths[row]] - 1).tolist(), history)
            self.assertTrue(torch.equal(lengths, expected_lengths))

    def test_model_scores_exact_catalog_shape(self):
        model = Mamba4Rec(self.data.num_items, hidden_size=8, mixer_factory=IdentityMixer)
        sequences, lengths, _ = evaluation_batch(self.data, sorted(self.data.test_target)[:3], "test", 4, "cpu")
        scores = model.full_catalog_scores(sequences, lengths)
        self.assertEqual(scores.shape, (3, self.data.num_items))
        self.assertTrue(torch.isfinite(model.loss(sequences, lengths, torch.tensor([0, 1, 2]))))

    def test_metrics_use_conservative_ties(self):
        scores = torch.ones(2, self.data.num_items)
        totals = ranking_metrics(scores, torch.tensor([0, 1]))
        expected = 2.0 if self.data.num_items <= 5 else 0.0
        self.assertEqual(totals["hit@5"], expected)

    def test_cache_identity_matches_shared_cache(self):
        args = Namespace(dataset="amazon-all-beauty", data_path=None, max_events=None,
                         min_rating=4.0, cache_dir="/tmp/shared-cache")
        self.assertEqual(interaction_cache_path(args).parent.name, "mamba_multi_agent_data")


if __name__ == "__main__": unittest.main()
