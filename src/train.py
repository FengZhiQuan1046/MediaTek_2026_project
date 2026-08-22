from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.data import build_data, edge_index, load_amazon, synthetic_events
from src.model import HybridRecommender, load_or_encode_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    """Set all local random generators before loading data or creating a model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(log_dir: Path, run_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recommender")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_dir / f"train_{run_id}.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_distributed(enabled: bool) -> tuple[int, int, int]:
    """Initialise one NCCL process per selected GPU when launched by torchrun."""
    if not enabled:
        return 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("--distributed requires CUDA GPUs.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def save_loss_curve(losses: list[float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", label="BPR loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training loss", xticks=range(1, len(losses) + 1))
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_test_samples(model, data, edges, device, path: Path, count: int = 5) -> None:
    """Save a small, human-readable set of held-out ranking examples as JSON."""
    model.eval()
    rows = []
    for user in sorted(data.test_target)[:count]:
        target = data.test_target[user]
        user_tensor = torch.tensor([user], device=device)
        all_items = torch.arange(data.num_items, device=device).unsqueeze(0)
        scores = model.score(user_tensor, all_items, edges, padded_histories(data, user_tensor, device))[0]
        top_ids = torch.topk(scores, k=min(10, data.num_items)).indices.tolist()
        rows.append({
            "user_index": user,
            "history_item_indices": data.train_by_user[user],
            "held_out_item_index": target,
            "held_out_item_text": data.item_texts[target],
            "top_recommendations": [
                {"item_index": item, "item_text": data.item_texts[item], "score": round(float(scores[item]), 6)}
                for item in top_ids
            ],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def padded_histories(data, users, device):
    histories = [data.train_by_user[int(u)] for u in users.tolist()]
    width = max(map(len, histories))
    # Left-pad with the first observed item; GRU uses the last actual item either way.
    return torch.tensor([[h[0]] * (width - len(h)) + h for h in histories], device=device)


def candidates_with_target(targets, num_items, negatives, device):
    candidates = []
    for target in targets.tolist():
        pool = [i for i in range(num_items) if i != target]
        sampled = random.sample(pool, k=min(negatives, len(pool)))
        candidates.append([target] + sampled)
    return torch.tensor(candidates, device=device)


@torch.no_grad()
def evaluate(model, data, edges, targets, device, k=10):
    model.eval()
    users = torch.tensor(sorted(targets), device=device)
    gold = torch.tensor([targets[int(u)] for u in users.tolist()], device=device)
    candidates = candidates_with_target(gold, data.num_items, 99, device)
    scores = model.score(users, candidates, edges, padded_histories(data, users, device))
    ranks = (scores[:, 1:] >= scores[:, :1]).sum(1) + 1
    recall = (ranks <= k).float().mean().item()
    ndcg = torch.where(ranks <= k, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)).mean().item()
    return recall, ndcg


def main():
    p = argparse.ArgumentParser(description="Mamba + bipartite graph recommender demo")
    p.add_argument("--subset", default="raw_review_All_Beauty")
    p.add_argument("--max-events", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=str(PROJECT_ROOT / "cache"), help="Hugging Face dataset and Mamba cache directory")
    p.add_argument("--artifact", default=None, help="Optional item-vector cache path; defaults under --cache-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default=str(PROJECT_ROOT / "log"))
    p.add_argument("--fig-output-dir", default=str(PROJECT_ROOT / "fig_outputs"))
    p.add_argument("--json-output-dir", default=str(PROJECT_ROOT / "json_outputs"))
    p.add_argument("--test-sample-count", type=int, default=5)
    p.add_argument("--distributed", action="store_true", help="Enable DDP; set by run.sh for two or more GPUs")
    p.add_argument("--skip-mamba", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    rank, world_size, local_rank = setup_distributed(args.distributed)
    is_main_process = rank == 0
    if args.distributed:
        args.device = f"cuda:{local_rank}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(Path(args.log_dir), run_id) if is_main_process else logging.getLogger("recommender.worker")
    if not is_main_process:
        logger.addHandler(logging.NullHandler())
    seed_everything(args.seed)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact or str(cache_dir / f"mamba_item_vectors_{args.subset}.pt")
    logger.info("run_id=%s seed=%d device=%s cache_dir=%s world_size=%d", run_id, args.seed, args.device, cache_dir, world_size)

    events = synthetic_events() if args.synthetic else load_amazon(args.subset, args.max_events, str(cache_dir))
    data = build_data(events)
    logger.info("users=%d items=%d train_edges=%d", data.num_users, data.num_items, sum(map(len, data.train_by_user.values())))
    # Only rank 0 ever loads the 2.8B model for encoding. Other ranks wait, then
    # read the completed item-vector cache, avoiding duplicate model memory.
    if is_main_process:
        text = load_or_encode_text(data.item_texts, artifact, args.device, args.skip_mamba, str(cache_dir))
    if args.distributed:
        dist.barrier()
    if not is_main_process:
        text = load_or_encode_text(data.item_texts, artifact, args.device, args.skip_mamba, str(cache_dir))
    # Restore identical initial recommender weights across DDP ranks after rank 0's encoding.
    seed_everything(args.seed)
    model = HybridRecommender(data.num_users, data.num_items, args.dim, text).to(args.device)
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    edges = edge_index(data).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    train_users = list(data.train_by_user)

    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_users); losses = []
        local_users = train_users[rank::world_size]
        # DDP requires every rank to execute the same number of backward calls.
        # Pad the final shard by repeating local users, rather than dropping data.
        if args.distributed:
            if not local_users:
                raise RuntimeError("Number of training users must be at least the number of selected GPUs.")
            batches_per_rank = math.ceil(len(train_users) / (world_size * args.batch_size))
            required_users = batches_per_rank * args.batch_size
            repeats = math.ceil(required_users / len(local_users))
            local_users = (local_users * repeats)[:required_users]
        batch_starts = range(0, len(local_users), args.batch_size)
        progress = tqdm(
            batch_starts,
            total=math.ceil(len(local_users) / args.batch_size),
            disable=not is_main_process,
            dynamic_ncols=True,
            desc=f"Epoch {epoch}/{args.epochs} | BPR ranking",
            unit="step",
        )
        for step, start in enumerate(progress, start=1):
            users = torch.tensor(local_users[start:start + args.batch_size], device=args.device)
            positive = torch.tensor([random.choice(data.train_by_user[int(u)]) for u in users.tolist()], device=args.device)
            negative = torch.randint(data.num_items, positive.shape, device=args.device)
            # BPR needs a genuinely unobserved comparison item for each draw.
            while torch.any(negative == positive):
                clash = negative == positive
                negative[clash] = torch.randint(data.num_items, (int(clash.sum()),), device=args.device)
            candidates = torch.stack((positive, negative), 1)
            scores = model(users, candidates, edges, padded_histories(data, users, args.device))
            loss = F.softplus(-(scores[:, 0] - scores[:, 1])).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
            if is_main_process:
                progress.set_postfix_str(
                    f"step={step}/{progress.total} | backward+optimizer | loss={loss.item():.5f}"
                )
        loss_stats = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=args.device)
        if args.distributed:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        average_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
        if is_main_process:
            epoch_losses.append(average_loss)
            recall, ndcg = evaluate(unwrap_model(model), data, edges, data.valid_target, args.device)
            logger.info("epoch=%d loss=%.4f valid Recall@10=%.4f NDCG@10=%.4f", epoch, average_loss, recall, ndcg)
        if args.distributed:
            dist.barrier()
    if is_main_process:
        recommender = unwrap_model(model)
        recall, ndcg = evaluate(recommender, data, edges, data.test_target, args.device)
        logger.info("TEST Recall@10=%.4f NDCG@10=%.4f", recall, ndcg)
        loss_path = Path(args.fig_output_dir) / f"loss_{run_id}.png"
        samples_path = Path(args.json_output_dir) / f"test_samples_{run_id}.json"
        save_loss_curve(epoch_losses, loss_path)
        save_test_samples(recommender, data, edges, args.device, samples_path, args.test_sample_count)
        logger.info("loss curve: %s", loss_path)
        logger.info("test samples: %s", samples_path)
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
