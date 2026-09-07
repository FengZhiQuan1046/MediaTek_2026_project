"""Train Mamba4Rec with MediaTek ver4 preprocessing and evaluation."""
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

from data_adapter import (evaluation_batch, load_data_cached, mask_seen_items,
                          ranking_metrics, sample_prefix_batch)
from model import LargeMamba4Rec, MAMBA_BACKEND, Mamba4Rec


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(path):
    logger = logging.getLogger("mamba4rec_reproduction")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def amp_context(device, enabled, dtype=torch.float16):
    return torch.amp.autocast("cuda", dtype=dtype) if enabled and device.startswith("cuda") else nullcontext()


def clip_gradients(model, maximum):
    """Clip each device shard without gathering multi-GPU gradients."""
    parameters_by_device = {}
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameters_by_device.setdefault(parameter.grad.device, []).append(parameter)
    for parameters in parameters_by_device.values():
        torch.nn.utils.clip_grad_norm_(parameters, maximum)


@torch.inference_mode()
def evaluate(model, data, split, args):
    model.eval()
    targets = data.valid_target if split == "valid" else data.test_target
    users = sorted(targets)
    if not users:
        raise RuntimeError(f"No eligible users exist in the {split} split")
    totals = {f"{metric}@{k}": 0.0 for metric in ("recall", "hit", "ndcg") for k in (5, 10)}
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), args.eval_batch_size), desc=f"full-catalog {split}"):
        batch_users = users[start:start + args.eval_batch_size]
        sequences, lengths, histories = evaluation_batch(data, batch_users, split, args.maxlen, args.device)
        amp_dtype = (
            torch.bfloat16
            if args.use_large_mamba and args.large_model_dtype == "bfloat16"
            else torch.float16
        )
        with amp_context(args.device, args.amp, amp_dtype):
            scores = model.full_catalog_scores(sequences, lengths).float()
        gold = torch.tensor([targets[user] for user in batch_users], device=args.device)
        mask_seen_items(scores, histories, gold)
        for name, value in ranking_metrics(scores, gold).items():
            totals[name] += value
    seconds = max(time.perf_counter() - started, 1e-9)
    metrics = {name: value / len(users) for name, value in totals.items()}
    metrics.update({"evaluated_users": len(users), "users_per_second": len(users) / seconds,
                    "scores_per_second": len(users) * data.num_items / seconds})
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Mamba4Rec on the exact MediaTek ver4 protocol")
    parser.add_argument("--dataset", required=True); parser.add_argument("--data-path")
    parser.add_argument("--cache-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refresh-data-cache", action="store_true")
    parser.add_argument("--max-events", type=int); parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--maxlen", type=int, default=100); parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d-state", type=int, default=32); parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2); parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0); parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--monitor-metric", choices=("ndcg@10", "recall@10"), default="ndcg@10")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0); parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-large-mamba", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--large-model-id", default="state-spaces/mamba-2.8b-hf")
    parser.add_argument("--large-model-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--large-device-map", choices=("none", "auto", "balanced"), default="none")
    args = parser.parse_args()
    positive = (args.maxlen, args.hidden_size, args.num_layers, args.d_state, args.d_conv,
                args.expand, args.epochs, args.batch_size, args.eval_batch_size)
    if min(positive) < 1 or args.early_stopping_patience < 0 or args.max_batches_per_epoch < 0:
        parser.error("dimensions, epochs and batch sizes must be positive; limits must be non-negative")
    return args


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(output / f"train_{run_id}.log")
    logger.info("EXPERIMENT_CONFIG %s", json.dumps(vars(args), sort_keys=True))
    logger.info(
        "MAMBA_BACKEND %s",
        args.large_model_id if args.use_large_mamba else MAMBA_BACKEND,
    )
    logger.info("LARGE_DEVICE_MAP %s", args.large_device_map)
    data, cache = load_data_cached(args, logger)
    logger.info("dataset=%s users=%d evaluation_users=%d items=%d train_interactions=%d cache=%s",
                args.dataset, data.num_users, len(data.test_target), data.num_items,
                sum(map(len, data.train_by_user.values())), cache)
    if args.use_large_mamba:
        model = LargeMamba4Rec(
            data.num_items,
            args.large_model_id,
            args.cache_dir,
            args.large_model_dtype,
            None if args.large_device_map == "none" else args.large_device_map,
        )
        if args.large_device_map == "none":
            model = model.to(args.device)
    else:
        model = Mamba4Rec(data.num_items, args.hidden_size, args.num_layers, args.dropout,
                          args.d_state, args.d_conv, args.expand).to(args.device)
    logger.info("device=%s trainable_parameters=%d", args.device,
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and args.device.startswith("cuda")
    amp_dtype = (
        torch.bfloat16
        if args.use_large_mamba and args.large_model_dtype == "bfloat16"
        else torch.float16
    )
    # BF16 has the same exponent range as FP32 and must not use FP16 gradient
    # scaling; PyTorch's CUDA unscale kernel is intentionally unavailable for
    # BF16 gradients.
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    logger.info(
        "MIXED_PRECISION enabled=%s dtype=%s grad_scaler=%s",
        use_amp,
        str(amp_dtype).removeprefix("torch."),
        use_grad_scaler,
    )
    batches = max(1, math.ceil(len(data.train_by_user) / args.batch_size))
    if args.max_batches_per_epoch:
        batches = min(batches, args.max_batches_per_epoch)
    rng = random.Random(args.seed); best_score = -math.inf; best_state = None
    best_epoch = 0; best_valid = None; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        progress = tqdm(range(batches), desc=f"Mamba4Rec {epoch}/{args.epochs}")
        for _ in progress:
            sequences, lengths, targets = sample_prefix_batch(data, args.batch_size, args.maxlen, rng, args.device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, use_amp, amp_dtype):
                loss = model.loss(sequences, lengths, targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            clip_gradients(model, args.grad_clip)
            scaler.step(optimizer); scaler.update(); total += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step(); valid = evaluate(model, data, "valid", args)
        average = total / batches; score = float(valid[args.monitor_metric])
        history.append({"epoch": epoch, "loss": average, "valid": valid})
        logger.info("epoch=%d loss=%.6f valid=%s monitor=%s score=%.6f",
                    epoch, average, json.dumps(valid, sort_keys=True), args.monitor_metric, score)
        if score > best_score:
            best_score, best_epoch, best_valid, stale = score, epoch, valid, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                logger.info("EARLY_STOP epoch=%d best_epoch=%d", epoch, best_epoch); break
    model.load_state_dict(best_state)
    test = evaluate(model, data, "test", args)
    result = {"dataset": args.dataset, "best_epoch": best_epoch, "monitor_metric": args.monitor_metric,
              "best_score": best_score, "validation": best_valid, "test": test,
              "num_users": data.num_users, "evaluation_users": len(data.test_target),
              "num_items": data.num_items, "train_interactions": sum(map(len, data.train_by_user.values())),
              "visible_gpus": torch.cuda.device_count() if args.device.startswith("cuda") else 0, "history": history}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("FINAL_RESULT %s", json.dumps(result, sort_keys=True)); logger.info("outputs=%s", output)


if __name__ == "__main__":
    main()
