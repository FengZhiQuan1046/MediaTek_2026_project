"""Checks for candidate ranking metrics and terminal-only progress output."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranking import ranking_metrics  # noqa: E402


class RankingMetricsTest(unittest.TestCase):
    def test_one_relevant_item_at_various_ranks(self):
        metrics = ranking_metrics([1, 5, 6, 10, 11])
        self.assertAlmostEqual(metrics["recall@5"], 2 / 5)
        self.assertAlmostEqual(metrics["recall@10"], 4 / 5)
        self.assertAlmostEqual(
            metrics["ndcg@5"], (1 + 1 / math.log2(6)) / 5
        )
        self.assertAlmostEqual(
            metrics["ndcg@10"],
            (1 + 1 / math.log2(6) + 1 / math.log2(7) + 1 / math.log2(11)) / 5,
        )


class ProgressOutputTest(unittest.TestCase):
    def test_progress_stays_on_terminal_and_out_of_log(self):
        output = (
            b"Loading LLAMA Done\n"
            b"\rEpoch 0:  50%|#####| 2/4 [00:01<00:01]\r"
            b"Generating full split: 123 examples [00:01, 123 examples/s]\n"
            b"\x1b[AValidation DataLoader 0:  25%|##| 1/4 [00:01<00:03]\n"
            b"warning: keep this\nTraceback: keep this too\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            result = subprocess.run(
                [sys.executable, str(ROOT / "filter_progress.py"), str(log)],
                input=output, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout, output)
            self.assertEqual(
                log.read_bytes(),
                b"Loading LLAMA Done\nwarning: keep this\nTraceback: keep this too\n",
            )


if __name__ == "__main__":
    unittest.main()
