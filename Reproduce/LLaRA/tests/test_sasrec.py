"""The bundled recommender must train and reload without upstream modules."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_rec import train_rec_model  # noqa: E402
from sasrec import SASRec  # noqa: E402


class LocalSASRecTest(unittest.TestCase):
    def test_pretraining_checkpoint_is_safe_and_usable(self):
        data = SimpleNamespace(num_items=5, train_by_user={0: [0, 1, 2], 1: [1, 3, 4]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rec.pt"
            train_rec_model(data, path, "cpu", epochs=1, batch_size=2,
                            hidden_size=8, maxlen=4, seed=1)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["version"], 1)
            model = SASRec(checkpoint["hidden_size"], checkpoint["item_num"],
                            checkpoint["state_size"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                scores = model(torch.tensor([[0, 1, 5, 5]]), torch.tensor([2]))
                embeddings = model.cacu_x(torch.tensor([[0, 1, 5]]))
            self.assertEqual(tuple(scores.shape), (1, 5))
            self.assertEqual(tuple(embeddings.shape), (1, 3, 8))
            self.assertTrue(torch.isfinite(scores).all().item())


if __name__ == "__main__":
    unittest.main()
