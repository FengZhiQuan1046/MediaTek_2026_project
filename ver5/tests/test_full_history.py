"""Regression tests for preference inputs longer than the old 100-event cap."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from src.model_mamba_rl import MultiAgentMambaRecommender
from src.train_mamba_rl import Transition, history_batch, evaluation_history_batch, parse_args, sequence_input_limit


class FullHistoryTest(unittest.TestCase):
    def setUp(self):
        self.data=SimpleNamespace(train_by_user={0:list(range(170)),1:[2,3,4]},
                                  valid_target={0:170,1:5},test_target={0:171,1:6})

    def test_training_full_prefix_and_explicit_cap(self):
        batch,lengths,targets=history_batch(self.data,[Transition(0,137,137)],0,'cpu')
        self.assertEqual(batch[0].tolist(),list(range(137)))
        self.assertEqual(lengths.tolist(),[137])
        self.assertEqual(targets.tolist(),[137])
        batch,lengths,_=history_batch(self.data,[Transition(0,137,137)],100,'cpu')
        self.assertEqual(batch[0].tolist(),list(range(37,137)))
        self.assertEqual(lengths.tolist(),[100])

    def test_full_evaluation_without_future_targets_and_reversal(self):
        for split,size,short_size in [('valid',170,3),('test',171,4)]:
            batch,lengths,raw=evaluation_history_batch(self.data,[0,1],split,0,'cpu')
            self.assertEqual(raw[0],list(range(size)))
            self.assertEqual(batch[0].tolist(),raw[0])
            self.assertEqual(lengths.tolist(),[size,short_size])
            self.assertNotIn(171,raw[0])
        self.data.reverse_history_input=True
        batch,_,raw=evaluation_history_batch(self.data,[0],'test',0,'cpu')
        self.assertEqual(batch[0].tolist(),list(range(171))[::-1])
        self.assertEqual(raw[0],list(range(171)))

    def test_preference_over_100_short_window_and_backward(self):
        model=MultiAgentMambaRecommender(torch.randn(180,8),dim=8,preference_count=4,
                use_graph_embeddings=False,short_window=4,lora_dropout=0)
        self.assertEqual(sequence_input_limit(model,0),0)
        batch,lengths,_=history_batch(self.data,[Transition(0,137,137)],0,'cpu')
        with patch.object(model.preference_agent,'encode',wraps=model.preference_agent.encode) as pref:
            with patch.object(model.short_agent,'encode',wraps=model.short_agent.encode) as short:
                result=model(batch,lengths,torch.tensor([[137,175]]))
                self.assertEqual(pref.call_args.args[1].tolist(),[137])
                self.assertEqual(short.call_args.args[1].tolist(),[4])
                result['coordinator'].sum().backward()
                model.full_catalog_scores(batch,lengths)
                self.assertEqual(pref.call_args.args[1].tolist(),[137])
        model.use_preference=False
        self.assertEqual(sequence_input_limit(model,0),4)

    def test_default_unlimited(self):
        with patch('sys.argv',['train']):
            self.assertEqual(parse_args().max_history,0)
