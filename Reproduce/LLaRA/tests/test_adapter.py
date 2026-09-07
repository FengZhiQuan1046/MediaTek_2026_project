from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VER4 = ROOT.parents[1] / "ver4"
for path in (ROOT, VER4): sys.path.insert(0, str(path))
from amazon_data import AmazonData  # noqa: E402
from data_adapter import make_examples  # noqa: E402
from src.data import build_data, synthetic_events  # noqa: E402


class LLaRAAdapterTest(unittest.TestCase):
    def setUp(self): self.data = build_data(synthetic_events())

    def test_splits_are_ver4_boundaries(self):
        valid = make_examples(self.data, "valid", 10, 0, 1)
        test = make_examples(self.data, "test", 10, 0, 1)
        for user, history, target in valid:
            self.assertEqual(history, self.data.train_by_user[user])
            self.assertEqual(target, self.data.valid_target[user])
        for user, history, target in test:
            self.assertEqual(history, self.data.train_by_user[user] + [self.data.valid_target[user]])
            self.assertEqual(target, self.data.test_target[user])

    def test_candidates_contain_target_once(self):
        dataset = AmazonData(self.data, "test", cans_num=1, maxlen=10)
        sample = dataset[0]
        self.assertEqual(sample["cans"].count(sample["item_id"]), 1)
        self.assertEqual(sample["len_cans"], 1)


if __name__ == "__main__": unittest.main()
