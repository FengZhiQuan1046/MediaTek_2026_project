"""RQ5 analytical checks and tiny end-to-end CPU training/report validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src import rq5
from src.data import InteractionData
from src.model_mamba_rl import MultiAgentMambaRecommender

class RQ5Tests(unittest.TestCase):
    def test_entropy_and_ties(self):
        e = rq5.entropy_from_scores(torch.tensor([[100., 0., 0.], [0., 100., -100.]]), [[0], [0]])
        self.assertAlmostEqual(float(e[0]), 1.)
        self.assertLess(float(e[1]), .001)
        self.assertEqual(rq5.assign_groups([.2, .3, .4], [.3, .3]).tolist(), [0, 0, 2])

    def test_metrics_and_bootstrap(self):
        m = rq5.rank_metrics([1, 5, 6, 10, 11])
        np.testing.assert_array_equal(m[:, 2:], [[1,1],[1,1],[0,1],[0,1],[0,0]])
        self.assertAlmostEqual(m[1, 0], 1 / np.log2(6))
        np.testing.assert_allclose(rq5.bootstrap_means([.25]*6, 20, np.random.default_rng(5)), .25)
        self.assertIsNone(rq5.bootstrap_means([], 20, np.random.default_rng(5)))

    def test_rho_intervention(self):
        torch.manual_seed(7)
        model = MultiAgentMambaRecommender(torch.randn(20,8), dim=8, lora_rank=2,
            preference_count=4, use_graph_embeddings=False, short_window=2).eval()
        states = model.encode_states(torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([3,3]))
        same = rq5.coordinator_with_rho(model, states, states[7])
        torch.testing.assert_close(states[1], same[1])
        torch.testing.assert_close(states[3], same[3])
        with torch.no_grad(): model.coordinator.preference_context_gate.weight[0,0] = 2.
        original = rq5.coordinator_with_rho(model, states, states[7])
        changed = rq5.coordinator_with_rho(model, states, torch.zeros(2))
        for i in [0,2,4,5,6]: self.assertIs(states[i], changed[i])
        self.assertFalse(torch.allclose(original[3], changed[3]))

    def test_training_to_json(self):
        torch.set_num_threads(1)
        n, items = 12, 40
        data = InteractionData(
            train_by_user={u:[(u+i)%items for i in range(3+u%4)] for u in range(n)},
            valid_target={u:(u+12)%items for u in range(n)},
            test_target={u:(u+17)%items for u in range(n)},
            item_texts=[f'item {i}' for i in range(items)], num_users=n, num_items=items)
        with tempfile.TemporaryDirectory(prefix='rq5_test_') as tmp:
            out = Path(tmp)/'result'
            argv = ['rq5','--subset','Full_Beauty','--output-dir',str(out),'--device','cpu',
                '--epochs','1','--batch-size','4','--eval-batch-size','3','--max-transitions','8',
                '--shuffle-repeats','2','--bootstrap','20','--skip-mamba']
            evaluate, history = rq5.train.evaluate, rq5.train.evaluation_history_batch
            splits = []
            def record_history(*a, **kw):
                splits.append(a[2]); return history(*a, **kw)
            with patch.object(sys,'argv',argv), \
                 patch.object(rq5.train,'load_recommendation_data_cached',return_value=(data,Path(tmp)/'data.pkl')), \
                 patch.object(rq5.train,'evaluation_history_batch',side_effect=record_history):
                rq5.main()
            self.assertIs(rq5.train.evaluate, evaluate)
            self.assertEqual(set(splits), {'valid','test'})
            report = json.loads((out/'rq5.json').read_text())
            rows = [r for g in rq5.GROUPS for r in report['groups'][g]]
            self.assertEqual(len(rows), n)
            self.assertEqual(len({r['user_id'] for r in rows}), n)
            for r in rows:
                self.assertAlmostEqual(r['short_weight']+r['preference_weight'],1.,places=5)
                self.assertTrue(set(rq5.METRICS)<=set(r))
                self.assertEqual(len(r['rho_shuffled']['preference_weights']),2)
            with np.load(out/'rq5_arrays.npz') as a:
                for mode,key in [('full','full_rank'),('short_only','short_rank')]:
                    saved=json.loads((out/mode/'scores.json').read_text())
                    for metric,value in zip(rq5.METRICS,rq5.rank_metrics(a[key]).mean(0)):
                        self.assertAlmostEqual(saved['test'][metric],value,places=6)
                for perm in a['permutations']:
                    self.assertEqual(sorted(perm.tolist()),list(range(n)))
            for mode in ['full','short_only']:
                self.assertFalse((out/mode/'multi_agent_lora.pt').exists())
                self.assertFalse((out/mode/'best_validation.pt').exists())

if __name__=='__main__': unittest.main()
