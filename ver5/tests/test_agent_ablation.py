from contextlib import ExitStack
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.model_mamba_rl import MultiAgentMambaRecommender
from src.train_mamba_rl import Transition, history_batch, sequence_input_limit


class AgentAblationTest(unittest.TestCase):
    def test_all_two_agent_combinations(self):
        for use_short, use_preference, use_gcn in itertools.product((False, True), repeat=3):
            if not (use_short or use_preference):
                continue
            with self.subTest(flags=(use_short, use_preference, use_gcn)), ExitStack() as stack:
                model = MultiAgentMambaRecommender(
                    torch.randn(12, 8), dim=8, preference_count=4,
                    graph_edges=torch.tensor([[0, 1], [2, 3]]), graph_users=2,
                    use_graph_embeddings=use_gcn, use_short=use_short,
                    use_preference=use_preference,
                )
                self.assertFalse(hasattr(model, 'long_agent'))
                for name in ('short', 'preference'):
                    if not getattr(model, f'use_{name}'):
                        agent = getattr(model, f'{name}_agent')
                        stack.enter_context(patch.object(agent, 'encode', side_effect=AssertionError('disabled encoder called')))
                        stack.enter_context(patch.object(agent, 'logits', side_effect=AssertionError('disabled logits called')))
                if not use_gcn:
                    self.assertFalse(hasattr(model, 'graph'))
                for stage in ('specialists', 'joint', 'all_dl'):
                    model.set_stage(stage)
                    for name in ('short', 'preference'):
                        self.assertEqual(
                            any(p.requires_grad for p in getattr(model, f'{name}_agent').parameters()),
                            getattr(model, f'use_{name}'),
                        )
                    if model.coordinator is not None:
                        self.assertEqual(any(p.requires_grad for p in model.coordinator.parameters()), stage != 'specialists')
                    if use_gcn:
                        self.assertTrue(any(p.requires_grad for p in model.graph.parameters()))
                    output = model(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 2]), torch.tensor([[5, 6], [6, 7]]))
                    self.assertTrue(torch.isfinite(output['coordinator']).all())
                    self.assertNotIn('long', output)
                    self.assertEqual(set(model.adapter_routes()),
                                     {name for name, enabled in (('short', use_short), ('preference', use_preference),
                                                                  ('coordinator', model.use_coordinator)) if enabled})
                    if model.active_agent_count == 1:
                        self.assertIsNone(model.coordinator)
                        name = 'short' if use_short else 'preference'
                        self.assertTrue(torch.equal(output['coordinator'], output[name]))
                    output['coordinator'].sum().backward()
                    for name in ('short', 'preference'):
                        if not getattr(model, f'use_{name}'):
                            self.assertTrue(all(not p.requires_grad and p.grad is None
                                                for p in getattr(model, f'{name}_agent').parameters()))
                    if not use_short:
                        self.assertTrue((output['weights'][:, 0] == 0).all())
                    if not use_preference:
                        self.assertTrue((output['weights'][:, 1] == 0).all())
                        self.assertTrue((output['preference_weight'] == 0).all())
                    model.zero_grad(set_to_none=True)

    def test_preference_reads_full_history_and_short_reads_suffix(self):
        histories = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 4, 5, 0]])
        lengths = torch.tensor([5, 4])
        candidates = torch.tensor([[8, 9], [8, 9]])
        for use_short in (False, True):
            with self.subTest(use_short=use_short):
                model = MultiAgentMambaRecommender(
                    torch.randn(20, 8), dim=8, preference_count=4,
                    use_graph_embeddings=False, use_short=use_short,
                    use_preference=True, short_window=2, lora_dropout=0,
                )
                self.assertEqual(sequence_input_limit(model, 100), 100)
                data = SimpleNamespace(train_by_user={0: [1, 2, 3, 4, 5]})
                batch, batch_lengths, _ = history_batch(
                    data, [Transition(0, 5, 8)], sequence_input_limit(model, 100), 'cpu'
                )
                self.assertEqual(batch[0].tolist(), [1, 2, 3, 4, 5])
                self.assertEqual(batch_lengths.tolist(), [5])
                with patch.object(model.preference_agent, 'encode', wraps=model.preference_agent.encode) as preference_encode:
                    with patch.object(model.short_agent, 'encode', wraps=model.short_agent.encode) as short_encode:
                        model(histories, lengths, candidates)
                        self.assertEqual(preference_encode.call_args.args[0].shape[:2], (2, 5))
                        self.assertEqual(preference_encode.call_args.args[1].tolist(), [5, 4])
                        if use_short:
                            self.assertEqual(short_encode.call_args.args[0].shape[:2], (2, 2))
                            self.assertEqual(short_encode.call_args.args[1].tolist(), [2, 2])
                        else:
                            short_encode.assert_not_called()
                        model.full_catalog_scores(histories, lengths)
                        self.assertEqual(preference_encode.call_args.args[1].tolist(), [5, 4])

    def test_disabling_coordinator_uses_fixed_mixture_without_training_it(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 8), dim=8, preference_count=4,
            use_graph_embeddings=False, use_short=True, use_preference=True,
            use_coordinator=False, preference_score_weight=0.3,
        )
        self.assertIsNone(model.coordinator)
        self.assertNotIn('coordinator', model.adapter_routes())
        model.set_stage('joint')
        output = model(torch.tensor([[1, 2, 3]]), torch.tensor([3]), torch.tensor([[4, 5, 6]]))
        expected = 0.7 * output['short'] + 0.3 * output['preference']
        self.assertTrue(torch.allclose(output['coordinator'], expected))
        output['coordinator'].sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.short_agent.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.preference_agent.parameters()))

    def test_short_only_ignores_old_prefix_for_scores_and_gradients(self):
        model = MultiAgentMambaRecommender(
            torch.randn(20, 8), dim=8, preference_count=4,
            use_graph_embeddings=False, use_preference=False,
            short_window=2, lora_dropout=0,
        )
        self.assertEqual(sequence_input_limit(model, 100), 2)
        model.set_stage('joint')
        candidates = torch.tensor([[8, 9], [8, 9]])
        histories = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 4, 5, 0]])
        lengths = torch.tensor([5, 4])
        with patch.object(model.short_agent, 'encode', wraps=model.short_agent.encode) as encode:
            full = model(histories, lengths, candidates)
            self.assertEqual(encode.call_args.args[0].size(1), 2)
            self.assertEqual(encode.call_args.args[1].tolist(), [2, 2])
        full['coordinator'].sum().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        recent = model(torch.tensor([[4, 5], [4, 5]]), torch.tensor([2, 2]), candidates)
        recent['coordinator'].sum().backward()
        self.assertTrue(torch.equal(full['coordinator'], recent['coordinator']))
        for name, parameter in model.named_parameters():
            if name in gradients:
                self.assertTrue(torch.equal(gradients[name], parameter.grad), name)

    def test_graph_only_uses_user_item_dot_products(self):
        model = MultiAgentMambaRecommender(
            torch.randn(12, 8), dim=8, graph_edges=torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]),
            graph_users=2, use_graph_embeddings=True, use_short=False, use_preference=False,
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
        self.assertTrue(all(not p.requires_grad for name, p in model.named_parameters()
                            if not name.startswith('graph.')))

    def test_no_agent_or_graph_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiAgentMambaRecommender(torch.randn(12, 8), use_graph_embeddings=False,
                                      use_short=False, use_preference=False)
