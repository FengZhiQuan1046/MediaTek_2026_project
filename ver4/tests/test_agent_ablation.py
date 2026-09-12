from contextlib import ExitStack
import itertools
import unittest
from unittest.mock import patch
import torch
from src.model_mamba_rl import MultiAgentMambaRecommender


class AgentAblationTest(unittest.TestCase):
    def test_all_nonempty_combinations(self):
        for use_long, use_short, use_preference, use_gcn in itertools.product((False, True), repeat=4):
            if not (use_long or use_short or use_preference):
                continue
            with self.subTest(flags=(use_long, use_short, use_preference, use_gcn)), ExitStack() as stack:
                model = MultiAgentMambaRecommender(
                    torch.randn(12, 8), dim=8, preference_count=4, preference_hidden=8,
                    graph_edges=torch.tensor([[0, 1], [2, 3]]), graph_users=2,
                    use_graph_embeddings=use_gcn, use_long=use_long,
                    use_short=use_short, use_preference=use_preference,
                )
                for name in ('long', 'short', 'preference'):
                    if not getattr(model, f'use_{name}'):
                        agent = getattr(model, f'{name}_agent')
                        stack.enter_context(patch.object(agent, 'encode', side_effect=AssertionError('disabled encoder called')))
                        stack.enter_context(patch.object(agent, 'logits', side_effect=AssertionError('disabled logits called')))
                if not use_preference and model.coordinator is not None:
                    stack.enter_context(patch.object(model.coordinator.preference_context_gate, 'forward', side_effect=AssertionError('disabled preference gate called')))
                if model.coordinator is not None and not (use_long and use_short):
                    stack.enter_context(patch.object(model.coordinator.mix_lora, 'forward', side_effect=AssertionError('inactive mixing policy called')))
                if not use_gcn:
                    self.assertFalse(hasattr(model, 'graph'))
                for stage in ('specialists', 'coordinator', 'joint'):
                    if stage == 'coordinator' and not model.use_coordinator:
                        continue
                    model.set_stage(stage)
                    output = model(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 2]), torch.tensor([[5, 6], [6, 7]]))
                    self.assertTrue(torch.isfinite(output['coordinator']).all())
                    if model.active_agent_count == 1:
                        self.assertIsNone(model.coordinator)
                        name = 'long' if use_long else 'short' if use_short else 'preference'
                        self.assertTrue(torch.equal(output['coordinator'], output[name]))
                    output['coordinator'].sum().backward()
                    for name in ('long', 'short', 'preference'):
                        if not getattr(model, f'use_{name}'):
                            self.assertTrue(all(not p.requires_grad and p.grad is None for p in getattr(model, f'{name}_agent').parameters()))
                    if not use_long:
                        self.assertTrue((output['weights'][:, 0] == 0).all())
                    if not use_short:
                        self.assertTrue((output['weights'][:, 1] == 0).all())
                    if model.coordinator is not None and not (use_long and use_short):
                        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.coordinator.mix_lora.parameters()))
                    if not use_preference:
                        self.assertTrue((output['preference_weight'] == 0).all())
                    model.zero_grad(set_to_none=True)

    def test_no_long_ignores_old_prefix_for_scores_and_gradients(self):
        model = MultiAgentMambaRecommender(
            torch.randn(20, 8), dim=8, preference_count=4, preference_hidden=8,
            use_graph_embeddings=False, use_long=False, short_window=2, lora_dropout=0,
        )
        model.set_stage('joint')
        candidates = torch.tensor([[8, 9], [8, 9]])
        histories = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 4, 5, 0]])
        lengths = torch.tensor([5, 4])
        with patch.object(model.preference_agent, 'encode', wraps=model.preference_agent.encode) as encode:
            full = model(histories, lengths, candidates)
            self.assertEqual(encode.call_args.args[0].size(1), 2)
            self.assertEqual(encode.call_args.args[1].tolist(), [2, 2])
        full['coordinator'].sum().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        recent = model(torch.tensor([[4, 5], [4, 5]]), torch.tensor([2, 2]), candidates)
        recent['coordinator'].sum().backward()
        self.assertTrue(torch.equal(full['coordinator'], recent['coordinator']))
        for name, p in model.named_parameters():
            if name in gradients:
                self.assertTrue(torch.equal(gradients[name], p.grad), name)

    def test_graph_only_uses_user_item_dot_products(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 8), dim=8, graph_edges=torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]),
            graph_users=2, use_graph_embeddings=True, use_long=False, use_short=False, use_preference=False,
        )
        model.set_stage('joint')
        self.assertIsNone(model.coordinator)
        items = torch.tensor([[4, 5], [6, 7]])
        user_ids = torch.tensor([0, 1])
        output = model(torch.tensor([[1, 2], [2, 3]]), torch.tensor([2, 2]), items, user_ids=user_ids)
        users, vectors = model.graph(model.graph_edges)
        expected = (users[user_ids].unsqueeze(1) * vectors[items]).sum(-1)
        self.assertTrue(torch.allclose(expected, output['coordinator']))
        output['coordinator'].sum().backward()
        self.assertIsNotNone(model.graph.embedding.weight.grad)
        self.assertTrue(all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith('graph.')))

    def test_no_specialist_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiAgentMambaRecommender(torch.randn(12, 8), use_graph_embeddings=False,
                                      use_long=False, use_short=False, use_preference=False)
