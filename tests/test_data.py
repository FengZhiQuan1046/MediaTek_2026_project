import importlib.util
import unittest

from src.data import (
    _amazon_positive, _deduplicate_user_items, build_data, edge_index, synthetic_events,
)


class DataTest(unittest.TestCase):
    def test_amazon_positive_filter_and_parent_item_deduplication(self):
        self.assertTrue(_amazon_positive({"rating": 4.0}, 4.0))
        self.assertFalse(_amazon_positive({"rating": 3.0}, 4.0))
        self.assertFalse(_amazon_positive({}, 4.0))
        rows = [
            ("u1", "parent-a", 10, "old"),
            ("u1", "parent-a", 20, "new"),
            ("u1", "parent-b", 15, "other"),
        ]
        deduplicated = sorted(_deduplicate_user_items(rows), key=lambda row: row[1])
        self.assertEqual(deduplicated, [("u1", "parent-a", 20, "new"), ("u1", "parent-b", 15, "other")])

    def test_chronological_split(self):
        data = build_data(synthetic_events())
        self.assertEqual(data.num_users, 24)
        self.assertTrue(all(len(history) == 3 for history in data.train_by_user.values()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_edges(self):
        data = build_data(synthetic_events())
        edges = edge_index(data)
        self.assertEqual(edges.shape[0], 2)
        self.assertGreater(edges.shape[1], 0)
