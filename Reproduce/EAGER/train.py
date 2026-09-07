"""EAGER reproduction on the exact MediaTek ver4 Amazon-2023 protocol."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parents[2] / "EAGER" / "EAGER"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from lib import DINTrain, Trm4Rec  # noqa: E402
from lib.KmeansTree import ConstructKmeansTree, kmeans_equal  # noqa: E402
from optimizers import AdamOptimizer  # noqa: E402
from optimizers.lr_schedulers import InverseSquareRootSchedule  # noqa: E402

from data_adapter import evaluation_tensors, load_data, training_tensors  # noqa: E402
from semantic_features import load_or_encode  # noqa: E402


def _device_safe_kmeans(self, index):
    """Upstream algorithm with its CUDA-mask/CPU-index device mismatch fixed."""
    cluster_size = len(index) // self.k
    with torch.no_grad():
        self.data = self.data.cuda()
        selected = torch.randperm(self.data.shape[1])[
            :math.ceil(self.feature_ratio * self.data.shape[1])
        ]
        choices, _ = kmeans_equal(
            self.data[:, selected][index], cluster_size=cluster_size,
            num_clusters=self.k, max_iters=self.max_iters,
        )
    result = torch.full((self.k, cluster_size), -1, dtype=torch.int64)
    for cluster in range(self.k):
        result[cluster] = index[(choices == cluster).cpu()]
    self.data = self.data.cpu()
    return result


ConstructKmeansTree._kmeans = _device_safe_kmeans


def configure_logging(path):
    logger = logging.getLogger("eager_reproduction")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cycle_batches(histories, labels, batch_size):
    while True:
        yield from DataLoader(
            TensorDataset(histories, labels), batch_size=batch_size, shuffle=True,
            drop_last=False, num_workers=0,
        )


def train_din(args, data, histories, labels, cache, logger):
    feature_groups = [5, 4, 2, 2, 1, 1, 1, 1, 1, 1]
    trainer = DINTrain(
        item_num=data.num_items, sample_negative_num=args.din_negatives,
        emb_dim=96, device=args.device, feature_groups=feature_groups,
        sum_pooling=False,
        optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-3, amsgrad=True),
    )
    if cache.exists():
        trainer.DINModel.load_state_dict(torch.load(cache, map_location=args.device, weights_only=True))
        logger.info("DIN_CACHE hit path=%s", cache)
        return trainer.DINModel
    iterator = cycle_batches(histories, labels, args.din_batch_size)
    trainer.DINModel.train()
    losses = []
    for step in tqdm(range(1, args.din_steps + 1), desc="DIN pretraining", unit="step"):
        batch_x, batch_y = next(iterator)
        loss = trainer.update_DIN(batch_x, batch_y)
        losses.append(float(loss.detach()))
        if step % args.log_every_steps == 0:
            logger.info("DIN step=%d loss=%.6f", step, float(np.mean(losses[-args.log_every_steps:])))
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.DINModel.state_dict(), cache)
    logger.info("DIN_CACHE saved path=%s", cache)
    return trainer.DINModel


def build_streams(args, data, behavior_features, semantic_features, cache_root):
    k = args.branch_factor or math.ceil(math.sqrt(data.num_items))
    # Upstream equal-size k-means processes points in groups of four.
    if args.branch_factor == 0 and k % 2:
        k += 1
    if k * k < data.num_items:
        raise ValueError("branch_factor must satisfy branch_factor**2 >= num_items")
    if k % 2:
        raise ValueError("branch_factor must be even for upstream EAGER k-means")
    optimizer = lambda parameters: torch.optim.Adam(parameters, lr=1e-3, amsgrad=True)
    streams = []
    feature_sets = (behavior_features.cpu(), semantic_features.cpu())
    dimensions = (96, int(semantic_features.size(1)))
    for stream_id, (features, dimension) in enumerate(zip(feature_sets, dimensions)):
        item_to_code = cache_root / f"stream{stream_id}_k{k}_item_to_code.npy"
        code_to_item = cache_root / f"stream{stream_id}_k{k}_code_to_item.npy"
        cached = item_to_code.exists() and code_to_item.exists()
        cache_root.mkdir(parents=True, exist_ok=True)
        stream = Trm4Rec(
            item_num=data.num_items, user_seq_len=args.maxlen, d_model=args.d_model,
            d_model2=dimension, nhead=args.num_heads, device=args.device,
            optimizer=optimizer, enc_num_layers=1, dec_num_layers=2, k=k,
            item_to_code_file=str(item_to_code), code_to_item_file=str(code_to_item),
            tree_has_generated=cached, init_way="embkm", max_iters=args.kmeans_iters,
            feature_ratio=1.0, data=features, parall=args.tree_workers,
            type=stream_id,
        )
        if streams:
            stream.trm_model.trm.encoder = streams[0].trm_model.trm.encoder
        streams.append(stream)
    return streams, feature_sets, k


def rerank(streams, histories, candidates, topk):
    rows = []
    for candidate_row in candidates:
        rows.append(list(dict.fromkeys(int(item) for item in candidate_row)))
    width = max(map(len, rows))
    labels = torch.zeros((len(rows), width), dtype=torch.long, device=histories.device)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for row, values in enumerate(rows):
        labels[row, :len(values)] = torch.tensor(values, device=histories.device)
        valid[row, :len(values)] = True
    expanded_history = histories.repeat_interleave(width, dim=0)
    expanded_items = labels.reshape(-1)
    scores = torch.zeros_like(labels, dtype=torch.float32).masked_fill(~valid, -torch.inf)
    for stream_id, stream in enumerate(streams):
        scores += stream.compute_scores(
            expanded_history, expanded_items, type=stream_id
        ).sum(-1).view(len(rows), width).masked_fill(~valid, 0.0)
    indices = scores.topk(min(topk, width), dim=-1).indices
    return labels.gather(1, indices)


@torch.inference_mode()
def evaluate(streams, data, split, args, collect=False):
    for stream in streams: stream.trm_model.eval()
    users, histories, labels, raw_histories = evaluation_tensors(
        data, split, args.maxlen, args.eval_user_limit
    )
    totals = {f"{metric}@{cutoff}": 0.0 for metric in ("recall", "hit", "ndcg") for cutoff in (5, 10)}
    recommendations = []
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), args.eval_batch_size), desc=f"EAGER {split}", unit="batch"):
        batch_x = histories[start:start + args.eval_batch_size].to(args.device)
        generated = [
            stream.predict(batch_x, topk=args.predict_candidates, type=stream_id)
            for stream_id, stream in enumerate(streams)
        ]
        ranked = rerank(streams, batch_x, torch.cat(generated, dim=-1), args.predict_candidates)
        for row in range(len(batch_x)):
            seen = set(raw_histories[start + row])
            filtered = []
            for item in ranked[row].tolist():
                if item not in seen and item not in filtered:
                    filtered.append(item)
                if len(filtered) == 10: break
            gold = int(labels[start + row])
            for cutoff in (5, 10):
                selected = filtered[:cutoff]
                if gold in selected:
                    rank = selected.index(gold) + 1
                    totals[f"recall@{cutoff}"] += 1
                    totals[f"hit@{cutoff}"] += 1
                    totals[f"ndcg@{cutoff}"] += 1 / math.log2(rank + 1)
            if collect:
                recommendations.append({"user": users[start + row], "target": gold, "top10": filtered})
    seconds = max(time.perf_counter() - started, 1e-9)
    metrics = {name: value / len(users) for name, value in totals.items()}
    metrics.update({"evaluated_users": len(users), "users_per_second": len(users) / seconds})
    return metrics, recommendations


def train_eager(args, data, histories, labels, streams, features, logger):
    parameters = list(streams[0].trm_model.trm.encoder.parameters())
    for stream in streams:
        parameters += list(stream.trm_model.trm.decoder.parameters())
        parameters += list(stream.trm_model.fc_proj1.parameters())
        parameters += list(stream.trm_model.guide_proj.parameters())
        parameters += list(stream.trm_model.trans_d_rec.parameters())
        parameters += list(stream.trm_model.fc_comp.parameters())
        parameters += [stream.trm_model.start_vec, stream.trm_model.mask_vec]
    optimizer_args = {"lr": 1e-3, "weight_decay": 1e-7, "warmup_updates": 2000, "warmup_init_lr": 1e-7}
    optimizer = AdamOptimizer(optimizer_args, parameters)
    scheduler = InverseSquareRootSchedule(optimizer_args, optimizer)
    iterator = cycle_batches(histories, labels, args.batch_size)
    best_score, best_step, best_valid, best_states = -math.inf, 0, None, None
    history, recent = [], []
    for stream in streams: stream.trm_model.train()
    for step in tqdm(range(1, args.train_steps + 1), desc="EAGER training", unit="step"):
        batch_x, batch_y = next(iterator); optimizer.zero_grad(); loss = 0.0; guide = None
        for stream_id in range(len(streams) - 1, -1, -1):
            stream_loss, _, guide = streams[stream_id].update_model(
                batch_x, batch_y, features[stream_id], type=stream_id,
                use_con=args.use_contrastive, use_guide=args.use_guide, guide_feat=guide,
            )
            loss = loss + stream_loss
        loss.backward(); optimizer.step(); learning_rate = scheduler.step_update(step)
        recent.append(float(loss.detach()))
        if step % args.log_every_steps == 0:
            logger.info("TRAIN step=%d loss=%.6f lr=%.8g", step, np.mean(recent[-args.log_every_steps:]), learning_rate)
        if args.eval_every_steps and (step % args.eval_every_steps == 0 or step == args.train_steps):
            valid, _ = evaluate(streams, data, "valid", args)
            score = valid[args.monitor_metric]
            history.append({"step": step, "loss": float(np.mean(recent[-args.eval_every_steps:])), "valid": valid})
            logger.info("VALID step=%d %s=%.6f metrics=%s", step, args.monitor_metric, score, json.dumps(valid, sort_keys=True))
            if score > best_score:
                best_score, best_step, best_valid = score, step, valid
                best_states = [{name: value.detach().cpu().clone() for name, value in stream.trm_model.state_dict().items()} for stream in streams]
            for stream in streams: stream.trm_model.train()
    if best_states is None:
        valid, _ = evaluate(streams, data, "valid", args)
        best_score, best_step, best_valid = valid[args.monitor_metric], args.train_steps, valid
    else:
        for stream, state in zip(streams, best_states): stream.trm_model.load_state_dict(state)
    return history, best_step, best_score, best_valid


def parse_args():
    parser = argparse.ArgumentParser(description="EAGER on exact ver4 Amazon-2023 data")
    parser.add_argument("--dataset", required=True); parser.add_argument("--data-path")
    parser.add_argument("--cache-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int); parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--din-steps", type=int, default=30000); parser.add_argument("--train-steps", type=int, default=60000)
    parser.add_argument("--din-batch-size", type=int, default=128); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=50); parser.add_argument("--eval-every-steps", type=int, default=300)
    parser.add_argument("--log-every-steps", type=int, default=100); parser.add_argument("--maxlen", type=int, default=19)
    parser.add_argument("--min-history", type=int, default=4); parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--branch-factor", type=int, default=0); parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4); parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--tree-workers", type=int, default=1); parser.add_argument("--din-negatives", type=int, default=60)
    parser.add_argument("--semantic-model-id", default="google-t5/t5-base")
    parser.add_argument("--semantic-backend", choices=("t5", "hashed"), default="t5")
    parser.add_argument("--semantic-batch-size", type=int, default=32); parser.add_argument("--semantic-max-tokens", type=int, default=64)
    parser.add_argument("--eval-user-limit", type=int, default=0); parser.add_argument("--predict-candidates", type=int, default=50)
    parser.add_argument("--monitor-metric", choices=("ndcg@10", "recall@10"), default="ndcg@10")
    parser.add_argument("--use-contrastive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-guide", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=2024); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.maxlen != 19: parser.error("upstream EAGER DIN feature groups require --maxlen 19")
    if args.d_model != 96: parser.error("upstream EAGER auxiliary modules require --d-model 96")
    if args.d_model % args.num_heads: parser.error("d_model must be divisible by num_heads")
    if min(args.din_steps, args.train_steps, args.batch_size, args.din_batch_size, args.predict_candidates) < 1:
        parser.error("training steps, batch sizes and prediction candidates must be positive")
    return args


def main():
    args = parse_args(); seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA is required by upstream EAGER")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(output / f"train_{run_id}.log")
    logger.info("EXPERIMENT_CONFIG %s", json.dumps(vars(args), sort_keys=True))
    data, interaction_cache = load_data(args, logger)
    histories, labels = training_tensors(data, args.maxlen, args.min_history, args.max_train_samples, args.seed)
    safe = args.dataset.replace(":", "_").replace("/", "_")
    eager_cache = Path(args.cache_dir) / "eager" / safe
    din_cache = eager_cache / f"din_steps{args.din_steps}_seed{args.seed}.pt"
    din = train_din(args, data, histories, labels, din_cache, logger)
    behavior_features = din.item_embedding.embed.weight.detach()[:data.num_items].cpu()
    semantic, semantic_cache = load_or_encode(
        data.item_texts, args.cache_dir, args.dataset, args.semantic_model_id,
        args.semantic_batch_size, args.semantic_max_tokens, args.device, args.semantic_backend,
    )
    streams, features, branch_factor = build_streams(args, data, behavior_features, semantic, eager_cache / "trees")
    history, best_step, best_score, validation = train_eager(args, data, histories, labels, streams, features, logger)
    test, recommendations = evaluate(streams, data, "test", args, collect=True)
    score_path = output / f"{output.parent.name}_scores.json"
    score_path.write_text(json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "dataset": args.dataset, "method": "EAGER", "best_step": best_step,
        "monitor_metric": args.monitor_metric, "best_score": best_score,
        "validation": validation, "test": test, "num_users": data.num_users,
        "evaluation_users": len(data.test_target), "num_items": data.num_items,
        "train_interactions": sum(map(len, data.train_by_user.values())),
        "train_examples": len(histories), "branch_factor": branch_factor,
        "streams": [
            "DIN_behavior",
            "T5_semantic" if args.semantic_backend == "t5" else "hashed_semantic_smoke_test",
        ],
        "visible_gpus": torch.cuda.device_count(),
        "interaction_cache": str(interaction_cache), "din_cache": str(din_cache),
        "semantic_cache": str(semantic_cache), "tree_cache": str(eager_cache / "trees"),
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("FINAL_RESULT %s", json.dumps(result, sort_keys=True)); logger.info("outputs=%s", output)


if __name__ == "__main__":
    main()
