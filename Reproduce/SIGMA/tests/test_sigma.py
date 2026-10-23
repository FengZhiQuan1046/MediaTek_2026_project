import importlib.util
from pathlib import Path
import sys
import types
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import SIGMA, UpstreamSIGMA
from data_adapter import evaluation_batch


class SIGMATest(unittest.TestCase):
    def test_upstream_forward_loss_and_real_catalog_scores_are_preserved(self):
        config=dict(hidden_size=8,num_layers=1,dropout_prob=0.0,loss_type='CE',d_state=4,d_conv=4,expand=2)
        upstream=UpstreamSIGMA(config,types.SimpleNamespace(item_num=13)).eval()
        model=SIGMA(12,8,1,0.0,4,4,2).eval()
        model.load_state_dict(upstream.state_dict())
        seq=torch.tensor([[1,2,3],[4,5,0]])
        lengths=torch.tensor([3,2])
        target=torch.tensor([3,5])
        self.assertTrue(torch.equal(model(seq,lengths),upstream(seq,lengths)))
        interaction={'item_seq':seq,'item_seq_len':lengths,'pos_item_id':target+1}
        self.assertTrue(torch.equal(model.loss(seq,lengths,target),upstream.calculate_loss(interaction)))
        self.assertTrue(torch.equal(model.full_catalog_scores(seq,lengths),upstream.full_sort_predict(interaction)[:,1:]))
        model.loss(seq,lengths,target).backward()
        self.assertIsNotNone(model.item_embedding.weight.grad)

    def test_microbatch_gradient_matches_full_batch(self):
        from train import backward_microbatches
        torch.manual_seed(13)
        full=SIGMA(12,8,1,0.0,4,4,2).eval()
        micro=SIGMA(12,8,1,0.0,4,4,2).eval()
        micro.load_state_dict(full.state_dict())
        sequences=torch.tensor([[1,2,3],[4,5,6],[2,3,4]])
        lengths=torch.tensor([3,3,3])
        targets=torch.tensor([4,7,5])
        expected=full.loss(sequences,lengths,targets)
        expected.backward()
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        actual=backward_microbatches(micro,sequences,lengths,targets,2,scaler,'cpu',False)
        self.assertAlmostEqual(actual,float(expected.detach()),places=6)
        for (name,left),(_,right) in zip(full.named_parameters(),micro.named_parameters()):
            if left.grad is not None:
                self.assertTrue(torch.allclose(left.grad,right.grad,atol=2e-6,rtol=2e-5),name)

    def test_full_history_and_exact_validation_test_boundary(self):
        data=types.SimpleNamespace(train_by_user={0:list(range(130))},valid_target={0:130},test_target={0:131})
        seq,lengths,hist=evaluation_batch(data,[0],'valid',0,'cpu')
        self.assertEqual(hist[0],list(range(130)))
        self.assertEqual(seq[0].tolist(),list(range(1,131)))
        seq,lengths,hist=evaluation_batch(data,[0],'test',0,'cpu')
        self.assertEqual(hist[0],list(range(131)))
        self.assertEqual(lengths.tolist(),[131])
        model=SIGMA(132,8,1,0.0,4,4,2).eval()
        self.assertEqual(model.full_catalog_scores(seq,lengths).shape,(1,132))


if __name__=='__main__':unittest.main()
