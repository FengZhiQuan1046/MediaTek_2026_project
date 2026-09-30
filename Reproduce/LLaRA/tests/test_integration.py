"""Regression checks for the LLaRA reproduction boundary."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VER4 = ROOT.parents[1] / "ver4"
for path in (ROOT, VER4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from amazon_data import AmazonData
from src.data import build_data, synthetic_events
from train import ReproductionInterface


class LLaRAIntegrationTest(unittest.TestCase):
    def test_too_many_candidates_fails_instead_of_looping_forever(self):
        data = build_data(synthetic_events())
        dataset = AmazonData(data, "test", cans_num=10, maxlen=10)
        with self.assertRaisesRegex(ValueError, "available candidates"):
            dataset[0]

    def test_distributed_eval_deduplicates_padded_users(self):
        part = {
            "user": [1, 1, 2],
            "generate": ["a", "wrong duplicate", "b"],
            "real": ["a", "a", "b"],
            "cans": [["a"], ["a"], ["b"]],
        }
        merged = ReproductionInterface._merge_content(part)
        self.assertEqual(merged, {"generate": ["a", "b"], "real": ["a", "b"],
                                  "cans": [["a"], ["b"]]})


if __name__ == "__main__":
    unittest.main()
