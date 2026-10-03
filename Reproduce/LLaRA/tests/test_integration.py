"""Regression checks for full-catalog LLaRA scoring and ver4 ranking."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ranking import full_catalog_scores, rank_like_ver4  # noqa: E402


class LLaRAIntegrationTest(unittest.TestCase):
    def test_full_catalog_scores_match_independent_continuations(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32,
                                             intermediate_size=64, num_hidden_layers=1,
                                             num_attention_heads=4, num_key_value_heads=2))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                                                  target_modules=["q_proj", "v_proj"]))
        model.eval()
        prompt = model.get_input_embeddings()(torch.tensor([1, 2, 3]))
        ids = torch.tensor([[4, 5, 6], [7, 8, 0], [9, 10, 11]])
        masks = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]])
        actual = full_catalog_scores(model, ids, masks, prompt, chunk_size=2)
        expected = []
        for row in range(len(ids)):
            length = int(masks[row].sum())
            combined = torch.cat((torch.tensor([1, 2, 3]), ids[row, :length]))
            logits = model(input_ids=combined.unsqueeze(0)).logits.float()[0, 2:-1]
            log_probs = logits.log_softmax(-1)
            expected.append(log_probs.gather(1, ids[row, :length].unsqueeze(1)).mean())
        torch.testing.assert_close(actual, torch.stack(expected), atol=1e-5, rtol=1e-5)

    def test_ver4_priors_apply_before_seen_mask(self):
        priors = SimpleNamespace(popularity=torch.tensor([0.0, 0.0, 1.0, 0.0]),
                                 transitions={1: {3: 2.0}})
        scores = torch.tensor([0.0, 0.2, 0.3, 0.1])
        # Seen item 1 is removed, then the transition prior promotes item 3.
        self.assertEqual(rank_like_ver4(scores, [1], 2, priors,
                                        popularity_alpha=0.5, transition_beta=1.0), 2)
        self.assertEqual(rank_like_ver4(scores, [1], 3, priors,
                                        popularity_alpha=0.5, transition_beta=1.0), 1)

    def test_ver4_seen_mask_and_tie_rule(self):
        scores = torch.tensor([0.9, 0.8, 0.8, 0.1])
        self.assertEqual(rank_like_ver4(scores, [0], 1), 2)
        self.assertEqual(rank_like_ver4(scores, [0, 1], 1), 2)
        self.assertEqual(rank_like_ver4(scores, [0, 2], 1), 1)


if __name__ == "__main__":
    unittest.main()
