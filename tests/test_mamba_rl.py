import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.data_mamba_rl import amazon_subset, load_movielens, load_recommendation_data
    from src.model_mamba_rl import MultiAgentMambaRecommender
    from src.train_mamba_rl import build_transitions, candidate_slates, history_batch


@unittest.skipUnless(HAS_TORCH, "requires torch")
class MultiAgentMambaRLTest(unittest.TestCase):
    def test_agents_have_disjoint_lora_parameters_and_full_catalog_scores(self):
        model = MultiAgentMambaRecommender(torch.randn(12, 16), dim=8, lora_rank=2, short_window=2)
        histories = torch.tensor([[1, 2, 3], [3, 4, 0]])
        lengths = torch.tensor([3, 2])
        candidates = torch.tensor([[4, 5, 6], [5, 6, 7]])
        output = model(histories, lengths, candidates)
        self.assertEqual(output["coordinator"].shape, (2, 3))
        self.assertEqual(model.full_catalog_scores(histories, lengths)["coordinator"].shape, (2, 12))
        self.assertTrue(set(model.long_agent.parameters()).isdisjoint(set(model.short_agent.parameters())))

    def test_each_training_stage_selects_expected_parameters(self):
        model = MultiAgentMambaRecommender(torch.randn(8, 6), dim=4, lora_rank=2)
        model.set_stage("coordinator")
        self.assertTrue(all(parameter.requires_grad for parameter in model.coordinator.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.long_agent.parameters()))
        model.set_stage("joint")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_synthetic_transition_batch(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        transitions = build_transitions(data, maximum=10, seed=1)
        histories, lengths, targets = history_batch(data, transitions[:4], 10, "cpu")
        slates = candidate_slates(targets, data.num_items, 4)
        self.assertEqual(histories.size(0), 4)
        self.assertTrue(torch.all(lengths >= 1))
        self.assertTrue(torch.equal(slates[:, 0], targets))

    def test_amazon_category_config_name(self):
        self.assertEqual(amazon_subset("amazon-all-beauty"), "raw_review_All_Beauty")
        self.assertEqual(amazon_subset("amazon-movies-and-tv"), "raw_review_Movies_and_TV")

    def test_movielens_csv_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "movies.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("movieId", "title", "genres"))
                writer.writerow(("10", "A Film", "Drama"))
            with (root / "ratings.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("userId", "movieId", "rating", "timestamp"))
                writer.writerow(("u1", "10", "5", "123"))
                writer.writerow(("u1", "10", "2", "124"))
            rows = load_movielens("movielens-1m", str(root), directory, None, 4.0)
            self.assertEqual(rows, [("u1", "10", 123, "A Film Genres: Drama")])


if __name__ == "__main__":
    unittest.main()
