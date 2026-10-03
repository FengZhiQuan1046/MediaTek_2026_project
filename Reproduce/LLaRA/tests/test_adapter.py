from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VER4 = ROOT.parents[1] / "ver4"
for path in (ROOT, VER4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from amazon_data import AmazonData  # noqa: E402
from data_adapter import make_examples  # noqa: E402
from src.data import build_data, synthetic_events  # noqa: E402
from src.train_mamba_rl import build_transitions  # noqa: E402


class LLaRAAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_data(synthetic_events())

    def test_train_transition_selection_is_ver4(self):
        for maximum in (0, 10):
            expected = build_transitions(self.data, maximum, 25252)
            actual = make_examples(self.data, "train", 100, maximum, 25252)
            self.assertEqual(
                actual,
                [(row.user, self.data.train_by_user[row.user][:row.end], row.target)
                 for row in expected],
            )

    def test_splits_are_ver4_boundaries(self):
        valid = make_examples(self.data, "valid", 100, 0, 25252)
        test = make_examples(self.data, "test", 100, 0, 25252)
        self.assertEqual([user for user, _, _ in valid], sorted(self.data.valid_target))
        self.assertEqual([user for user, _, _ in test], sorted(self.data.test_target))
        for user, history, target in valid:
            self.assertEqual(history, self.data.train_by_user[user])
            self.assertEqual(target, self.data.valid_target[user])
        for user, history, target in test:
            self.assertEqual(history, self.data.train_by_user[user] + [self.data.valid_target[user]])
            self.assertEqual(target, self.data.test_target[user])

    def test_pilot_user_selection_is_ver4_even_spacing(self):
        expected = sorted(self.data.test_target)
        limit = 5
        users = [user for user, _, _ in make_examples(self.data, "test", 100, limit, 25252)]
        self.assertEqual(users, [expected[index * len(expected) // limit]
                                 for index in range(limit)])

    def test_dataset_has_no_sampled_candidates(self):
        dataset = AmazonData(self.data, "test", maxlen=100)
        sample = dataset[0]
        self.assertEqual(sample["item_id"], self.data.test_target[sample["user"]])
        self.assertNotIn("cans", sample)
        self.assertEqual(len(dataset), len(self.data.test_target))


if __name__ == "__main__":
    unittest.main()
