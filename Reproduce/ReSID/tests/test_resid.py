from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import random
import sys
import unittest

import torch


RESID_ROOT = Path(__file__).resolve().parents[1]
MEDIATEK_ROOT = RESID_ROOT.parents[1]
VER4_ROOT = MEDIATEK_ROOT / "ver4"
for path in (RESID_ROOT, VER4_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_adapter import (  # noqa: E402
    evaluation_histories,
    interaction_cache_path,
    mask_seen_items,
    pad_item_histories,
    ranking_metrics,
    sample_prefix_batch,
)
from models import (  # noqa: E402
    FieldAwareMaskedAutoEncoder,
    GenerativeSIDRecommender,
    build_item_fields,
    globally_aligned_orthogonal_quantization,
)
from src.data import MIN_INTERACTIONS, build_data, synthetic_events  # noqa: E402
from src.train_mamba_rl import evaluation_history_batch  # noqa: E402


class ExactProtocolTest(unittest.TestCase):
    def setUp(self):
        self.data = build_data(synthetic_events())

    def test_validation_and_test_histories_equal_ver4(self):
        users = sorted(self.data.test_target)[:7]
        for split in ("valid", "test"):
            actual_histories = evaluation_histories(self.data, users, split, max_len=2)
            actual_tensor = pad_item_histories(actual_histories, 2, "cpu")
            _, ver4_lengths, ver4_histories = evaluation_history_batch(
                self.data, users, split, max_history=2, device="cpu"
            )
            self.assertEqual(actual_histories, ver4_histories)
            expected_tensor = torch.tensor(
                [[item + 1 for item in history] for history in ver4_histories]
            )
            self.assertTrue(torch.equal(actual_tensor, expected_tensor))
            self.assertEqual(ver4_lengths.tolist(), [2] * len(users))

    def test_seen_mask_and_metrics_match_project_formula(self):
        histories = [[0, 1, 2], [1, 3]]
        gold = torch.tensor([3, 1])
        scores = torch.tensor(
            [[0.9, 0.8, 0.7, 1.0, 0.1], [0.1, 0.8, 0.4, 0.3, 0.2]]
        )
        masked = mask_seen_items(scores.clone(), histories, gold)
        self.assertTrue(torch.isneginf(masked[0, :3]).all())
        self.assertFalse(torch.isneginf(masked[1, 1]))
        ranks = (masked >= masked.gather(1, gold[:, None])).sum(1)
        metrics = ranking_metrics(masked, gold)
        expected_hits = float((ranks <= 5).sum())
        self.assertEqual(metrics["recall@5"], expected_hits)
        self.assertEqual(metrics["hit@5"], expected_hits)

    def test_training_sampler_never_uses_validation_or_test_targets(self):
        histories, targets = sample_prefix_batch(
            self.data, 64, 4, random.Random(7), "cpu"
        )
        for row in range(len(targets)):
            target = int(targets[row]) - 1
            self.assertTrue(
                any(target in sequence for sequence in self.data.train_by_user.values())
            )
        self.assertTrue(torch.all(histories >= 0))

    def test_cache_path_uses_common_project_identity(self):
        args = Namespace(
            dataset="amazon-toys-and-games",
            data_path=None,
            max_events=None,
            min_rating=4.0,
            cache_dir="/tmp/shared-cache",
        )

        # The path is asserted through the same identity formula and directory;
        # no I/O is needed for this check.
        expected_identity = {
            "dataset": args.dataset,
            "data_path": None,
            "max_events": None,
            "min_rating": 4.0,
            "min_interactions": MIN_INTERACTIONS,
            "schema_version": 2,
        }
        import hashlib
        import json

        digest = hashlib.sha1(
            json.dumps(expected_identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        expected = Path(args.cache_dir) / "mamba_multi_agent_data" / f"amazon-toys-and-games_{digest}.pkl"
        self.assertEqual(interaction_cache_path(args), expected)


class ReSIDModelTest(unittest.TestCase):
    def setUp(self):
        self.data = build_data(synthetic_events())

    def test_famae_gaoq_and_generator_shapes(self):
        fields, vocab_sizes = build_item_fields(
            self.data.item_texts, text_fields=2, text_buckets=16
        )
        histories = torch.tensor([[0, 1, 2], [0, 2, 3]])
        targets = torch.tensor([3, 4])
        famae = FieldAwareMaskedAutoEncoder(
            fields,
            vocab_sizes,
            max_len=3,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            item_candidates=8,
        )
        loss = famae.loss(histories, targets, [0, 1])
        self.assertTrue(torch.isfinite(loss))
        representations = famae.item_representations()
        codes = globally_aligned_orthogonal_quantization(
            representations, codebook1_size=3, codebook2_size=3
        )
        self.assertEqual(codes.shape, (self.data.num_items, 3))
        self.assertEqual(len(torch.unique(codes, dim=0)), self.data.num_items)

        model = GenerativeSIDRecommender(
            codes,
            max_len=3,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        logits = model(histories, targets)
        self.assertEqual(len(logits), 3)
        self.assertTrue(torch.isfinite(model.loss(histories, targets)))
        scores = model.score_all_items(histories, pair_chunk_size=2)
        self.assertEqual(scores.shape, (2, self.data.num_items))
        self.assertTrue(torch.isfinite(scores).all())
        for item_index in range(self.data.num_items):
            candidate = torch.full((2,), item_index + 1, dtype=torch.long)
            candidate_logits = model(histories, candidate)
            candidate_codes = codes[item_index] - 1
            expected = sum(
                torch.log_softmax(candidate_logits[level], dim=-1)[
                    :, candidate_codes[level]
                ]
                for level in range(3)
            )
            self.assertTrue(
                torch.allclose(scores[:, item_index], expected, atol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()
