from argparse import Namespace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VER4 = ROOT.parents[1] / "ver4"
for path in (ROOT, VER4): sys.path.insert(0, str(path))

from data_adapter import LLMRecData  # noqa: E402
from src.data import build_data, synthetic_events  # noqa: E402


class AdapterTest(unittest.TestCase):
    def test_exact_split_and_shapes(self):
        raw = build_data(synthetic_events())
        data = LLMRecData(raw, batch_size=8, feature_dim=16, seed=1)
        self.assertEqual(data.train_items, raw.train_by_user)
        self.assertEqual({u: x[0] for u, x in data.val_set.items()}, raw.valid_target)
        self.assertEqual({u: x[0] for u, x in data.test_set.items()}, raw.test_target)
        self.assertEqual(data.text_features.shape, (raw.num_items, 16))
        self.assertEqual(data.train_matrix.shape, (raw.num_users, raw.num_items))

    def test_sampler_uses_training_partition(self):
        raw = build_data(synthetic_events())
        data = LLMRecData(raw, batch_size=12, feature_dim=8, seed=2)
        users, positives, negatives = data.sample()
        for user, positive, negative in zip(users, positives, negatives):
            self.assertIn(positive, raw.train_by_user[user])
            self.assertNotIn(negative, raw.train_by_user[user])


if __name__ == "__main__": unittest.main()
