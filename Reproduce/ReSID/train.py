"""End-to-end ReSID reproduction on MediaTek's exact ver4 protocol."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from data_adapter import (
    evaluation_histories,
    load_data_cached,
    mask_seen_items,
    pad_item_histories,
    ranking_metrics,
    sample_prefix_batch,
)
from models import (
    FieldAwareMaskedAutoEncoder,
    GenerativeSIDRecommender,
    build_item_fields,
    globally_aligned_orthogonal_quantization,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("resid_reproduction")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(path, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def amp_context(device: str, enabled: bool):
    if enabled and device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_famae(args, data, item_fields, field_vocab_sizes, logger, rng):
    model = FieldAwareMaskedAutoEncoder(
        item_fields=item_fields.to(args.device),
        field_vocab_sizes=field_vocab_sizes,
        max_len=args.maxlen,
        hidden_size=args.hidden_size,
        num_layers=args.famae_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        item_candidates=args.famae_item_candidates,
    ).to(args.device)
    visible_gpus = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    logger.info(
        "device=%s visible_gpus=%d data_parallel=False trainable_parameters=%d",
        args.device,
        visible_gpus,
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.famae_learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.famae_epochs
    )
    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches = max(1, len(data.train_by_user) // args.batch_size)
    if args.max_batches_per_epoch:
        batches = min(batches, args.max_batches_per_epoch)
    for epoch in range(1, args.famae_epochs + 1):
        model.train()
        total = 0.0
        progress = tqdm(
            range(batches), desc=f"FAMAE epoch {epoch}/{args.famae_epochs}", unit="batch"
        )
        for _ in progress:
            item_history, targets = sample_prefix_batch(
                data, args.batch_size, args.maxlen, rng, args.device
            )
            number_masked = rng.randint(1, item_fields.size(1))
            masked_fields = rng.sample(range(item_fields.size(1)), number_masked)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, use_amp):
                loss = model.loss(item_history, targets, masked_fields)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step()
    return model


@torch.inference_mode()
def evaluate(model, data, split: str, args) -> dict[str, float]:
    """Exact project full-catalog evaluation with ReSID item scores."""
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
    for start in tqdm(
        range(0, len(users), args.eval_batch_size),
        desc=f"full-catalog {split}",
        unit="batch",
    ):
        batch_users = users[start : start + args.eval_batch_size]
        histories = evaluation_histories(data, batch_users, split, args.maxlen)
        item_histories = pad_item_histories(histories, args.maxlen, args.device)
        scores = model.score_all_items(
            item_histories, pair_chunk_size=args.score_pair_chunk_size
        )
        gold = torch.tensor([targets[user] for user in batch_users], device=args.device)
        mask_seen_items(scores, histories, gold)
        batch_totals = ranking_metrics(scores, gold)
        for name, value in batch_totals.items():
            totals[name] += value
    seconds = max(time.perf_counter() - started, 1e-9)
    metrics = {name: value / len(users) for name, value in totals.items()}
    metrics.update(
        {
            "evaluated_users": len(users),
            "users_per_second": len(users) / seconds,
            "scores_per_second": len(users) * data.num_items / seconds,
        }
    )
    return metrics


def train_recommender(args, data, item_codes, logger, rng):
    model = GenerativeSIDRecommender(
        item_codes=item_codes.to(args.device),
        max_len=args.maxlen,
        hidden_size=args.hidden_size,
        num_layers=args.recommender_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches = max(1, len(data.train_by_user) // args.batch_size)
    if args.max_batches_per_epoch:
        batches = min(batches, args.max_batches_per_epoch)

    best_score = -math.inf
    best_epoch = 0
    best_valid = None
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        progress = tqdm(
            range(batches), desc=f"ReSID epoch {epoch}/{args.epochs}", unit="batch"
        )
        for _ in progress:
            item_history, targets = sample_prefix_batch(
                data, args.batch_size, args.maxlen, rng, args.device
            )
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, use_amp):
                loss = model.loss(item_history, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step()

        valid_metrics = evaluate(model, data, "valid", args)
        average = total / batches
        result = {"epoch": epoch, "loss": average, "valid": valid_metrics}
        history.append(result)
        score = float(valid_metrics[args.monitor_metric])
        logger.info(
            "epoch=%d loss=%.6f valid=%s monitor=%s score=%.6f",
            epoch,
            average,
            json.dumps(valid_metrics, sort_keys=True),
            args.monitor_metric,
            score,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_valid = valid_metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if (
                args.early_stopping_patience
                and stale_epochs >= args.early_stopping_patience
            ):
                logger.info("EARLY_STOP epoch=%d best_epoch=%d", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_score, best_valid


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReSID reproduction using the exact MediaTek ver4 data protocol"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refresh-data-cache", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--famae-layers", type=int, default=2)
    parser.add_argument("--recommender-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-fields", type=int, default=4)
    parser.add_argument("--text-buckets", type=int, default=4096)
    parser.add_argument("--codebook1-size", type=int, default=64)
    parser.add_argument("--codebook2-size", type=int, default=64)
    parser.add_argument("--famae-epochs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--famae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--famae-item-candidates", type=int, default=4096)
    parser.add_argument("--score-pair-chunk-size", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument(
        "--monitor-metric", choices=("ndcg@10", "recall@10"), default="ndcg@10"
    )
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    positive = (
        args.maxlen,
        args.hidden_size,
        args.num_heads,
        args.famae_layers,
        args.recommender_layers,
        args.codebook1_size,
        args.codebook2_size,
        args.famae_epochs,
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.score_pair_chunk_size,
    )
    if min(positive) < 1:
        parser.error("model dimensions, epochs, batch sizes, and codebooks must be positive")
    if args.hidden_size % args.num_heads:
        parser.error("--hidden-size must be divisible by --num-heads")
    if args.codebook2_size > args.hidden_size:
        parser.error("--codebook2-size cannot exceed --hidden-size for orthogonal GAOQ")
    if args.max_batches_per_epoch < 0 or args.early_stopping_patience < 0:
        parser.error("stopping and batch limits must be non-negative")
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
    item_fields, field_vocab_sizes = build_item_fields(
        data.item_texts, args.text_fields, args.text_buckets
    )
    rng = random.Random(args.seed)
    famae = train_famae(args, data, item_fields, field_vocab_sizes, logger, rng)

    representations = famae.item_representations()
    item_codes = globally_aligned_orthogonal_quantization(
        representations,
        args.codebook1_size,
        args.codebook2_size,
        seed=args.seed,
    )
    del representations
    del famae
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    model, history, best_epoch, best_score, best_valid = train_recommender(
        args, data, item_codes, logger, rng
    )
    visible_gpus = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    test_metrics = evaluate(model, data, "test", args)
    result = {
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
