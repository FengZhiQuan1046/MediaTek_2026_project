"""Optional ranking aggregates using sequence lengths, never held-out item identities."""
import math


class SequenceLengthMetrics:
    def __init__(self, histories, targets, threshold, *, lengths=None, basis="untruncated_train_history"):
        self.threshold = threshold
        self.basis = basis
        self.lengths = {user: len(histories[user]) if lengths is None else lengths[user] for user in targets}
        self.groups = {}
        for name in ("short", "long"):
            lengths = [n for n in self.lengths.values() if self.group(n) == name]
            self.groups[name] = {
                "total_users": len(lengths), "evaluated_users": 0,
                "history_length_min": min(lengths) if lengths else None,
                "history_length_max": max(lengths) if lengths else None,
                "history_length_mean": sum(lengths) / len(lengths) if lengths else None,
                **{f"{metric}@{k}": 0.0 for metric in ("recall", "ndcg", "hit") for k in (5, 10)},
            }

    def group(self, length):
        return "short" if length <= self.threshold else "long"

    def update(self, users, ranks):
        for user, rank in zip(users, ranks):
            group = self.groups[self.group(self.lengths[user])]
            group["evaluated_users"] += 1
            for k in (5, 10):
                if rank <= k:
                    group[f"recall@{k}"] += 1
                    group[f"hit@{k}"] += 1
                    group[f"ndcg@{k}"] += 1 / math.log2(rank + 1)

    def report(self):
        groups = {}
        for name, totals in self.groups.items():
            count = totals["evaluated_users"]
            groups[name] = {
                key: (value / count if count else None) if "@" in key else value
                for key, value in totals.items()
            }
        return {
            "basis": self.basis, "threshold": self.threshold,
            "short_rule": "length <= threshold", "long_rule": "length > threshold",
            **groups,
        }


class PeriodicTestScores:
    """Persist each completed test immediately; keep history across final reporting."""
    def __init__(self, output, config, run_id):
        from pathlib import Path
        self.output = Path(output)
        self.config = config
        self.run_id = run_id
        self.history = {"short": [], "long": []}

    def record(self, metrics, stage, epoch, step):
        import json
        report = metrics["sequence_length"]
        for group in ("short", "long"):
            values = report[group]
            scores = {key: value for key, value in values.items() if "@" in key}
            self.history[group].append({
                "stage": stage, "epoch": epoch, "global_step": step,
                "scores": scores, "evaluated_users": values["evaluated_users"],
                "total_users": values["total_users"],
            })
            payload = {
                "dataset": self.config["dataset"], "run_id": self.run_id,
                "group": group, "length_basis": report["basis"],
                "length_threshold": report["threshold"], "length_rule": report[f"{group}_rule"],
                "config": self.config, "status": "training",
                "test": scores, "test_users": {
                    key: values[key] for key in ("evaluated_users", "total_users")
                },
                "periodic_test_history": self.history[group],
            }
            path = self.output / f"{group}_scores.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
