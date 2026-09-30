from pathlib import Path
from types import SimpleNamespace
import copy
import math
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import BERT4Rec
from train import MaskedSequenceDataset, evaluate, evaluation_sequences


def small_model(**kwargs):
    return BERT4Rec(num_items=12, maxlen=4, hidden_units=8, num_blocks=1,
                    num_heads=2, dropout_rate=0, **kwargs)


class BERT4RecTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.data = SimpleNamespace(
            num_items=12, train_by_user={0: [0, 1, 2], 1: [3]},
            valid_target={0: 4, 1: 5}, test_target={0: 6, 1: 7},
        )

    def test_masking_uses_only_training_targets_and_ignores_padding(self):
        dataset = MaskedSequenceDataset(self.data, maxlen=4, mask_prob=1)
        for index, history in enumerate(self.data.train_by_user.values()):
            sequence, labels = dataset[index]
            self.assertEqual(labels[labels != -100].tolist(), history)
            self.assertTrue(torch.all(labels[sequence == 0] == -100))
            self.assertTrue(torch.all((sequence >= 0) & (sequence <= 13)))
        # Even extremely low mask probability must produce a usable loss.
        sequence, labels = MaskedSequenceDataset(self.data, 4, 1e-12)[1]
        self.assertEqual(int((labels != -100).sum()), 1)
        self.assertEqual(int(labels[-1]), 3)

    def test_evaluation_appends_mask_without_leaking_targets(self):
        valid, histories = evaluation_sequences(self.data, [0], 'valid', 4, 'cpu')
        test, _ = evaluation_sequences(self.data, [0], 'test', 4, 'cpu')
        self.assertEqual(valid.tolist(), [[0, 1, 2, 3, 13]])
        self.assertEqual(test.tolist(), [[1, 2, 3, 5, 13]])
        self.assertEqual(histories, [[0, 1, 2]])
        truncated, _ = evaluation_sequences(self.data, [0], 'test', 2, 'cpu')
        self.assertEqual(truncated.tolist(), [[3, 5, 13]])

    def test_bidirectional_attention_and_padding_mask(self):
        model = small_model().eval()
        sequence = torch.tensor([[0, 1, 13, 3, 4]])
        before = model.encode(sequence).detach()
        changed = sequence.clone()
        changed[0, -1] = 5
        after = model.encode(changed).detach()
        self.assertFalse(torch.allclose(before[:, 1], after[:, 1], atol=1e-7))
        with torch.no_grad():
            model.item_embedding.weight[0].fill_(100)
        torch.testing.assert_close(before[:, 1:], model.encode(sequence)[:, 1:])
        self.assertEqual(model(sequence).shape, (1, 12))

    def test_chunked_mask_loss_and_gradients_match_full_softmax(self):
        model = small_model(loss_chunk_size=1).train()
        reference = copy.deepcopy(model)
        sequences = torch.tensor([[0, 1, 13, 3, 13], [0, 0, 0, 2, 13]])
        labels = torch.tensor([[-100, -100, 1, -100, 3], [-100, -100, -100, -100, 4]])
        sums, counts = model(sequences, labels)
        loss = sums.sum() / counts.sum()
        selected = labels != -100
        expected = F.cross_entropy(reference.scores(reference.encode(sequences)[selected]), labels[selected])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        expected.backward()
        for actual, target in zip(model.parameters(), reference.parameters()):
            if actual.grad is not None:
                torch.testing.assert_close(actual.grad, target.grad, atol=1e-6, rtol=1e-4)
        self.assertEqual(int(counts.sum()), 3)

    def test_full_catalog_seen_mask_target_preservation_and_ties(self):
        data = SimpleNamespace(num_items=12, train_by_user={0: [0, 1]},
                               valid_target={0: 1}, test_target={0: 2})
        class FixedScores(torch.nn.Module):
            def forward(self, sequence):
                scores = torch.zeros(len(sequence), 12)
                scores[:, 0] = 100  # seen: excluded
                scores[:, 1] = 2    # repeated target: must remain eligible
                scores[:, 2] = 2    # tie: pessimistic rank 2
                return scores
        metrics = evaluate(FixedScores(), data, 'valid', 1, 4, 'cpu')
        self.assertEqual(metrics['evaluated_users'], 1)
        self.assertAlmostEqual(metrics['ndcg@5'], 1 / math.log2(3), places=6)
        self.assertEqual(metrics['recall@5'], metrics['hit@5'])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires two CUDA GPUs')
    def test_data_parallel_unequal_mask_counts_and_backward(self):
        model = small_model(loss_chunk_size=1).cuda().train()
        reference = copy.deepcopy(model)
        parallel = torch.nn.DataParallel(model)
        sequences = torch.tensor([[0, 1, 13, 3, 13], [0, 0, 0, 2, 13]], device='cuda')
        labels = torch.tensor([[-100, -100, 1, -100, 3], [-100, -100, -100, -100, 4]], device='cuda')
        sums, counts = parallel(sequences, labels)
        loss = sums.sum() / counts.sum()
        expected_sums, expected_counts = reference(sequences, labels)
        expected = expected_sums.sum() / expected_counts.sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        expected.backward()
        torch.testing.assert_close(model.item_embedding.weight.grad,
                                   reference.item_embedding.weight.grad, atol=1e-6, rtol=1e-4)
        parallel.eval()
        reference.eval()
        with torch.inference_mode():
            torch.testing.assert_close(parallel(sequences), reference(sequences))


if __name__ == '__main__':
    unittest.main()
