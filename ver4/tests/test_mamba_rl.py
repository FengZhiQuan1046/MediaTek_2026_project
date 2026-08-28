import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.data_mamba_rl import amazon_item_group, amazon_subset, load_movielens, load_recommendation_data
    from src.model_mamba_rl import MultiAgentMambaRecommender
    from src.train_mamba_rl import (
        build_transitions, candidate_slates, evaluate, history_batch,
        preference_auxiliary_losses,
    )


@unittest.skipUnless(HAS_TORCH, "requires torch")
class MultiAgentMambaRLTest(unittest.TestCase):
    def test_agents_have_disjoint_lora_parameters_and_full_catalog_scores(self):
        edges = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        model = MultiAgentMambaRecommender(
            torch.randn(12, 16), dim=8, lora_rank=2, short_window=2,
            graph_edges=edges, graph_users=4,
        )
        histories = torch.tensor([[1, 2, 3], [3, 4, 0]])
        lengths = torch.tensor([3, 2])
        candidates = torch.tensor([[4, 5, 6], [5, 6, 7]])
        output = model(histories, lengths, candidates)
        self.assertEqual(output["coordinator"].shape, (2, 3))
        self.assertEqual(output["preference"].shape, (2, 3))
        self.assertEqual(output["preference_current"].shape, (2, 64))
        self.assertTrue(torch.allclose(output["preference_next"].sum(-1), torch.ones(2)))
        self.assertTrue(torch.all((output["preference_change"] >= 0) & (output["preference_change"] <= 1)))
        self.assertEqual(model.full_catalog_scores(histories, lengths)["coordinator"].shape, (2, 12))
        self.assertTrue(set(model.long_agent.parameters()).isdisjoint(set(model.short_agent.parameters())))

    def test_graph_embeddings_can_be_disabled(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 16), dim=8, lora_rank=2, use_graph_embeddings=False,
        )
        model.set_stage("specialists")
        model.set_stage("joint")
        histories = torch.tensor([[1, 2, 3]])
        lengths = torch.tensor([3])
        candidates = torch.tensor([[4, 5, 6]])
        self.assertEqual(model(histories, lengths, candidates)["coordinator"].shape, (1, 3))

    def test_each_training_stage_selects_expected_parameters(self):
        edges = torch.tensor([[0, 1], [4, 5]])
        model = MultiAgentMambaRecommender(torch.randn(8, 6), dim=4, lora_rank=2,
                                           graph_edges=edges, graph_users=4)
        model.set_stage("coordinator")
        self.assertTrue(all(parameter.requires_grad for parameter in model.coordinator.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.long_agent.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.graph.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.preference_agent.parameters()))
        model.set_stage("joint")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_preference_agent_has_finite_auxiliary_losses_and_gradients(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 8), dim=6, lora_rank=2, preference_count=4,
            preference_hidden=8, use_graph_embeddings=False,
        )
        model.set_stage("specialists")
        histories = torch.tensor([[1, 2, 3], [2, 4, 0]])
        lengths = torch.tensor([3, 2])
        candidates = torch.tensor([[4, 5, 6], [5, 6, 7]])
        vectors = model.project_ids(candidates)
        output = model.logits_from_states(model.encode_states(histories, lengths), vectors)
        losses = preference_auxiliary_losses(model, output, vectors[:, 0])
        loss = sum(losses) + torch.nn.functional.cross_entropy(
            output["preference"], torch.zeros(2, dtype=torch.long)
        )
        self.assertTrue(all(torch.isfinite(term) for term in losses))
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.preference_agent.parameters()
        ))

    def test_graph_embeddings_receive_joint_training_gradients(self):
        edges = torch.tensor([[0, 1], [4, 5]])
        model = MultiAgentMambaRecommender(torch.randn(8, 6), dim=4, lora_rank=2,
                                           graph_edges=edges, graph_users=4)
        model.set_stage("joint")
        output = model(
            torch.tensor([[1, 2, 3]]), torch.tensor([3]), torch.tensor([[4, 5, 6]]),
        )
        output["coordinator"].sum().backward()
        self.assertTrue(all(parameter.requires_grad for parameter in model.graph.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.graph.parameters()))

    def test_synthetic_transition_batch(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        transitions = build_transitions(data, maximum=10, seed=1)
        histories, lengths, targets = history_batch(data, transitions[:4], 10, "cpu")
        slates = candidate_slates(targets, data.num_items, 4)
        self.assertEqual(histories.size(0), 4)
        self.assertTrue(torch.all(lengths >= 1))
        self.assertTrue(torch.equal(slates[:, 0], targets))

    def test_transition_budget_prioritises_user_coverage(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        eligible_users = {user for user, history in data.train_by_user.items() if len(history) > 1}
        transitions = build_transitions(data, maximum=len(eligible_users), seed=7)
        self.assertEqual({row.user for row in transitions}, eligible_users)

    def test_amazon_category_config_name(self):
        self.assertEqual(amazon_subset("amazon-all-beauty"), "raw_review_All_Beauty")
        self.assertEqual(amazon_subset("amazon-movies-and-tv"), "raw_review_Movies_and_TV")
        self.assertEqual(amazon_subset("amazon-games"), "raw_review_Toys_and_Games")
        self.assertEqual(amazon_subset("amazon-toys"), "raw_review_Toys_and_Games")
        self.assertEqual(amazon_item_group("amazon-games"), "games")
        self.assertEqual(amazon_item_group("amazon-toys"), "toys")
        self.assertIsNone(amazon_item_group("amazon-toys-and-games"))

    def test_evaluation_reports_hit_at_5_and_10(self):
        data = load_recommendation_data("synthetic", None, "/tmp", min_user_events=5)
        model = MultiAgentMambaRecommender(
            torch.randn(data.num_items, 8), dim=4, lora_rank=2, use_graph_embeddings=False,
        )
        metrics, _ = evaluate(model, data, "test", 8, 10, "cpu")
        self.assertEqual(metrics["hit@5"], metrics["recall@5"])
        self.assertEqual(metrics["hit@10"], metrics["recall@10"])
        self.assertIn("preference_change_probability", metrics)

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
