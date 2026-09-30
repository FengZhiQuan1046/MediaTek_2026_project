"""Train and evaluate BERT4Rec with the exact ver4 data adapters and splits."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from itertools import islice
import logging
import math
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


BERT4REC_ROOT = Path(__file__).resolve().parent
MEDIATEK_ROOT = BERT4REC_ROOT.parents[1]
VER4_ROOT = MEDIATEK_ROOT / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))

from src.data import MIN_INTERACTIONS  # noqa: E402
from src.data_mamba_rl import load_recommendation_data  # noqa: E402
from model import BERT4Rec  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bert4rec_reproduction")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_data_cached(args, logger):
    """Use the same cache identity and normalized InteractionData as ver4."""
    identity = {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "max_events": args.max_events,
        "min_rating": args.min_rating,
        "min_interactions": MIN_INTERACTIONS,
        "schema_version": 2,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    safe_dataset = args.dataset.replace(":", "_").replace("/", "_")
    artifact = Path(args.cache_dir) / "mamba_multi_agent_data" / f"{safe_dataset}_{digest}.pkl"
    if artifact.exists() and not args.refresh_data_cache:
        with artifact.open("rb") as stream:
            data = pickle.load(stream)
        logger.info("INTERACTION_CACHE hit path=%s", artifact)
        return data, artifact

    data = load_recommendation_data(
        args.dataset,
        args.data_path,
        args.cache_dir,
        args.max_events,
        args.min_rating,
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(artifact)
    logger.info("INTERACTION_CACHE miss_saved path=%s", artifact)
    return data, artifact


class MaskedSequenceDataset(Dataset):
    """Fresh 80/10/10 corruption each epoch, using training interactions only."""
    def __init__(self, data, maxlen, mask_prob):
        self.histories = [history for history in data.train_by_user.values() if history]
        if not self.histories:
            raise RuntimeError("BERT4Rec needs at least one nonempty training history")
        self.num_items, self.maxlen, self.mask_prob = data.num_items, maxlen, mask_prob

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        items = torch.tensor(self.histories[index][-self.maxlen:], dtype=torch.long) + 1
        selected = torch.rand(len(items)) < self.mask_prob
        if not selected.any():
            selected[torch.randint(len(items), ())] = True
        labels = torch.full((self.maxlen + 1,), -100, dtype=torch.long)
        labels[-len(items):] = torch.where(selected, items - 1, -100)
        corruption = torch.rand(len(items))
        masked = selected & (corruption < 0.8)
        randomized = selected & (corruption >= 0.8) & (corruption < 0.9)
        items[masked] = self.num_items + 1
        items[randomized] = torch.randint(1, self.num_items + 1, (int(randomized.sum()),))
        sequence = torch.zeros(self.maxlen + 1, dtype=torch.long)
        sequence[-len(items):] = items
        return sequence, labels


def evaluation_sequences(data, users: list[int], split: str, maxlen: int, device: str):
    sequences = torch.zeros((len(users), maxlen + 1), dtype=torch.long, device=device)
    sequences[:, -1] = data.num_items + 1
    histories: list[list[int]] = []
    for row, user in enumerate(users):
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        history = history[-maxlen:]
        histories.append(history)
        if history:
            sequences[row, -len(history)-1:-1] = torch.tensor(
                [item + 1 for item in history], dtype=torch.long, device=device
            )
    return sequences, histories


@torch.inference_mode()
def evaluate(model, data, split: str, batch_size: int, maxlen: int, device: str):
    model.eval()
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    if not users:
        raise RuntimeError(f"No eligible users exist in the {split} split")
    totals = {
        "recall@5": 0.0,
        "recall@10": 0.0,
        "hit@5": 0.0,
        "hit@10": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
    }
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), batch_size), desc=f"full-catalog {split}", unit="batch"):
        batch_users = users[start:start + batch_size]
        sequences, histories = evaluation_sequences(data, batch_users, split, maxlen, device)
        scores = model(sequences).float()
        gold = torch.tensor([targets[user] for user in batch_users], device=device)
        for row, history in enumerate(histories):
            seen = set(history) - {int(gold[row])}
            if seen:
                scores[row, list(seen)] = -torch.inf
        ranks = (scores >= scores.gather(1, gold.unsqueeze(1))).sum(dim=1)
        for cutoff in (5, 10):
            hits = ranks <= cutoff
            hit_count = hits.sum().item()
            totals[f"recall@{cutoff}"] += hit_count
            totals[f"hit@{cutoff}"] += hit_count
            totals[f"ndcg@{cutoff}"] += torch.where(
                hits,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float),
            ).sum().item()
    seconds = max(time.perf_counter() - started, 1e-9)
    metrics = {name: value / len(users) for name, value in totals.items()}
    metrics.update({
        "evaluated_users": len(users),
        "users_per_second": len(users) / seconds,
        "scores_per_second": len(users) * data.num_items / seconds,
    })
    return metrics


def amp_context(device: str, enabled: bool):
    if enabled and device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(description="BERT4Rec reproduction using MediaTek ver4 data preprocessing")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refresh-data-cache", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mask-prob", type=float, default=0.2)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--monitor-metric", choices=("ndcg@10", "recall@10"), default="ndcg@10")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.maxlen, args.hidden_units, args.num_blocks, args.num_heads, args.batch_size, args.eval_batch_size) < 1:
        parser.error("model dimensions and batch sizes must be positive")
    if args.hidden_units % args.num_heads:
        parser.error("--hidden-units must be divisible by --num-heads")
    if args.epochs < 1 or args.early_stopping_patience < 0 or args.max_batches_per_epoch < 0:
        parser.error("epochs must be positive and stopping/batch limits non-negative")
    if not 0 < args.mask_prob <= 1:
        parser.error("--mask-prob must be in (0, 1]")
    if args.loss_chunk_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("loss chunk size and learning rate must be positive; weight decay non-negative")
    if not 0 <= args.dropout_rate < 1:
        parser.error("--dropout-rate must be in [0, 1)")
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive when supplied")
    return args


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(output / f"train_{run_id}.log")
    logger.info("EXPERIMENT_CONFIG %s", json.dumps(vars(args), sort_keys=True))
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    data, cache_artifact = load_data_cached(args, logger)
    logger.info(
        "dataset=%s users=%d evaluation_users=%d items=%d train_interactions=%d cache=%s",
        args.dataset,
        data.num_users,
        len(data.test_target),
        data.num_items,
        sum(map(len, data.train_by_user.values())),
        cache_artifact,
    )

    base_model = BERT4Rec(
        num_items=data.num_items,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        loss_chunk_size=args.loss_chunk_size,
    ).to(args.device)
    visible_gpus = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    model: torch.nn.Module = base_model
    if visible_gpus > 1:
        model = torch.nn.DataParallel(base_model, device_ids=list(range(visible_gpus)))
    logger.info(
        "device=%s visible_gpus=%d data_parallel=%s trainable_parameters=%d",
        args.device,
        visible_gpus,
        isinstance(model, torch.nn.DataParallel),
        sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dataset = MaskedSequenceDataset(data, args.maxlen, args.mask_prob)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=args.device.startswith("cuda"))
    batches_per_epoch = len(loader)
    if args.max_batches_per_epoch > 0:
        batches_per_epoch = min(batches_per_epoch, args.max_batches_per_epoch)

    best_score = -math.inf
    best_epoch = 0
    best_valid = None
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_masks = 0.0, 0
        progress = tqdm(islice(loader, batches_per_epoch), total=batches_per_epoch,
                        desc=f"train epoch {epoch}/{args.epochs}", unit="batch")
        for sequences, labels in progress:
            sequences = sequences.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, use_amp):
                loss_sums, counts = model(sequences, labels)
                loss = loss_sums.sum() / counts.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss_sums.detach().sum())
            total_masks += int(counts.sum())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        valid_metrics = evaluate(
            model, data, "valid", args.eval_batch_size, args.maxlen, args.device
        )
        epoch_result = {
            "epoch": epoch,
            "loss": total_loss / total_masks,
            "valid": valid_metrics,
        }
        history.append(epoch_result)
        score = float(valid_metrics[args.monitor_metric])
        logger.info(
            "epoch=%d loss=%.6f valid=%s monitor=%s score=%.6f",
            epoch,
            epoch_result["loss"],
            json.dumps(valid_metrics, sort_keys=True),
            args.monitor_metric,
            score,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_valid = valid_metrics
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in base_model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                logger.info("EARLY_STOP epoch=%d best_epoch=%d", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state")
    base_model.load_state_dict(best_state)
    test_metrics = evaluate(model, data, "test", args.eval_batch_size, args.maxlen, args.device)
    result = {
        "method": "BERT4Rec",
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "monitor_metric": args.monitor_metric,
        "best_score": best_score,
        "validation": best_valid,
        "test": test_metrics,
        "num_users": data.num_users,
        "evaluation_users": len(data.test_target),
        "num_items": data.num_items,
        "train_interactions": sum(map(len, data.train_by_user.values())),
        "visible_gpus": visible_gpus,
        "history": history,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("FINAL_RESULT %s", json.dumps(result, sort_keys=True))
    logger.info("outputs=%s", output)


if __name__ == "__main__":
    main()
