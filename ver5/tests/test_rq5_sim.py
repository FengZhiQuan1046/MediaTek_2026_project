"""Verify similarity screening, true full-history SLC and CPU end-to-end reports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from src import rq5_sim as sim
from src.data import InteractionData
from src.rq5_sim_model import SLCRecommender


class SimilarityTests(unittest.TestCase):
    def test_older_max_and_lags_are_distinct_from_preference(self):
        # Target0: item1 similar in text, item2 similar in preference.
        item = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        preference = torch.tensor([[1.,0.],[0.,1.],[1.,0.]])
        record = sim.history_similarities([1,2,2,2,2],0,item,preference,np.ones(3,bool),np.random.default_rng(2))
        self.assertGreater(record['item_gap'],0)
        self.assertLess(record['preference_gap'],0)
        self.assertEqual(record['item_older_lag'],5)
        reverse = sim.history_similarities([2,1,1,1,1],0,item,preference,np.ones(3,bool),np.random.default_rng(2))
        self.assertLess(reverse['item_gap'],0)
        self.assertGreater(reverse['preference_gap'],0)
        tie = sim.history_similarities([1,1,1,1,1],0,item,preference,np.ones(3,bool),np.random.default_rng(2))
        self.assertEqual(tie['item_gap'],0)
        self.assertIsNone(sim.history_similarities([1,2,1,2],0,item,preference,np.ones(3,bool),np.random.default_rng(2)))

    def test_proxy_fits_training_items_only(self):
        torch.manual_seed(3)
        features = torch.randn(40,8)
        data = InteractionData({0:list(range(32))},{0:35},{0:36},['x']*40,1,40)
        _,proxy,_,centers,ids = sim.fit_proxy(features,data,32,.2,40,5)
        self.assertEqual(ids.tolist(),list(range(32)))
        self.assertEqual(centers.shape,(32,8))
        torch.testing.assert_close(proxy.norm(dim=-1),torch.ones(40))

    def test_slc_reads_older_items_without_prototypes(self):
        torch.manual_seed(3)
        model=SLCRecommender(torch.randn(20,8),dim=8,lora_rank=2,preference_count=32,
            short_window=4,use_graph_embeddings=False).eval()
        histories=torch.tensor([[1,2,3,4,5,6],[7,8,3,4,5,6]])
        states=model.encode_states(histories,torch.tensor([6,6]))
        torch.testing.assert_close(states[0][0],states[0][1])
        self.assertFalse(torch.allclose(states[2][0],states[2][1]))
        self.assertFalse(any('prototypes' in name or 'next_head' in name or 'change_head' in name for name,_ in model.named_parameters()))
        self.assertIn('long',model.agent_parameter_counts())

    def test_end_to_end_spc_slc_and_json(self):
        torch.set_num_threads(1)
        n,items=12,48
        data=InteractionData({u:[(u*3+i)%items for i in range(7+u%3)] for u in range(n)},
            {u:(u*3+16)%items for u in range(n)}, {u:(u*3+19)%items for u in range(n)},
            [f'item{i}' for i in range(items)],n,items)
        original_class=sim.train.MultiAgentMambaRecommender
        original_aux=sim.train.preference_auxiliary_losses
        with tempfile.TemporaryDirectory(prefix='rq5_sim_test_') as temporary:
            out=Path(temporary)/'out'
            argv=['sim','--subset','Full_Beauty','--output-dir',str(out),'--device','cpu',
                '--epochs','1','--batch-size','4','--eval-batch-size','3','--max-transitions','8',
                '--bootstrap','20','--skip-mamba']
            with patch.object(sys,'argv',argv),patch.object(sim.train,'load_recommendation_data_cached',return_value=(data,Path(temporary)/'data.pkl')):
                sim.main()
            self.assertIs(sim.train.MultiAgentMambaRecommender,original_class)
            self.assertIs(sim.train.preference_auxiliary_losses,original_aux)
            report=json.loads((out/'summary.json').read_text())
            with np.load(out/'cohort_arrays.npz') as a:
                cohort_masks=sim.masks(a)
                self.assertTrue(np.array_equal(cohort_masks['intersection'],cohort_masks['item_similarity'] & cohort_masks['preference_similarity']))
                self.assertEqual(report['screening']['eligible_users'],n)
                for name in sim.COHORTS:
                    records=json.loads((out/f'{name}.json').read_text())['records']
                    self.assertEqual(len(records),int(cohort_masks[name].sum()))
                    for row in records:
                        self.assertIn('preference_weight',row['SPC'])
                        self.assertIn('long_weight',row['SLC'])
                        self.assertTrue(set(sim.rq5.METRICS)<=set(row['SPC']))
                for variant in ('SPC','SLC'):
                    scores=json.loads((out/variant/'scores.json').read_text())
                    actual=sim.rq5.rank_metrics(a[f'{variant}_rank']).mean(0)
                    for metric,value in zip(sim.rq5.METRICS,actual):
                        self.assertAlmostEqual(scores['test'][metric],value,places=6)
                    self.assertFalse((out/variant/'multi_agent_lora.pt').exists())
                    self.assertFalse((out/variant/'best_validation.pt').exists())
                self.assertEqual(json.loads((out/'SLC'/'scores.json').read_text())['model']['agents'],['short','long_item'])
                for idx in range(n):
                    self.assertEqual(a['history_length'][idx],len(data.train_by_user[idx])+1)


if __name__=='__main__': unittest.main()
