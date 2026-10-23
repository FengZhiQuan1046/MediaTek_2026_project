"""Durable, full-precision evaluation records for score-only runs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


class EvaluationWriter:
    def __init__(self, output, dataset):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.history = self.output / 'evaluations.jsonl'

    def write(self, label, metrics, stage, epoch, step):
        record = {
            'dataset': self.dataset,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'evaluation': label, 'stage': stage, 'epoch': epoch, 'step': step,
            'evaluated_users': metrics.get('evaluated_users'),
            'total_users': metrics.get('total_users'), 'metrics': metrics,
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.history.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temp = self.output / 'latest_evaluation.json.tmp'
        temp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(self.output / 'latest_evaluation.json')
