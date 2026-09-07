"""LLMRec training on ver4 data, retaining the upstream model and objectives."""
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
from torch.nn import functional as F
from tqdm.auto import tqdm

from data_adapter import LLMRecData, load_data, normalized_graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--aug-sample-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def logger_for(path):
    logger = logging.getLogger("llmrec_reproduction"); logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def import_upstream_model(args):
    upstream = Path(__file__).resolve().parents[3] / "LLMRec"
    sys.path.insert(0, str(upstream))
    original = sys.argv
    sys.argv = [original[0], "--embed_size", str(args.embed_size),
                "--batch_size", str(args.batch_size)]
    try:
        from Models import MM_Model, Decoder
    finally:
        sys.argv = original
    return MM_Model, Decoder


def prune_loss(prediction, drop_rate=0.71):
    indices = torch.argsort(prediction)
    remembered = max(1, int((1.0 - drop_rate) * len(indices)))
    return prediction[indices[:remembered]].mean()


def bpr_loss(users, positives, negatives, batch_size, decay=1e-5):
    positive_scores = (users * positives).sum(-1)
    negative_scores = (users * negatives).sum(-1)
    mf = -prune_loss(F.logsigmoid(positive_scores - negative_scores + 1e-8))
    regularizer = (
        1.0 / (2 * (users ** 2).sum() + 1e-8)
        + 1.0 / (2 * (positives ** 2).sum() + 1e-8)
        + 1.0 / (2 * (negatives ** 2).sum() + 1e-8)
    ) / batch_size
    return mf, decay * regularizer


@torch.inference_mode()
def evaluate(model, graphs, data, split, batch_size):
    model.eval()
    user_vectors, item_vectors, *_ = model(*graphs)
    targets = data.val_set if split == "valid" else data.test_set
    users = sorted(targets)
    totals = {f"{name}@{k}": 0.0 for name in ("recall", "hit", "ndcg") for k in (5, 10)}
    started = time.perf_counter()
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        scores = user_vectors[batch_users] @ item_vectors.T
        gold = torch.tensor([targets[u][0] for u in batch_users], device=scores.device)
        for row, user in enumerate(batch_users):
            history = list(data.train_items[user])
            if split == "test" and user in data.val_set:
                history.extend(data.val_set[user])
            seen = set(history) - {int(gold[row])}
            if seen:
                scores[row, list(seen)] = -torch.inf
        ranks = (scores >= scores.gather(1, gold[:, None])).sum(1)
        for cutoff in (5, 10):
            hits = ranks <= cutoff
            count = float(hits.sum())
            totals[f"recall@{cutoff}"] += count
            totals[f"hit@{cutoff}"] += count
            totals[f"ndcg@{cutoff}"] += float(torch.where(
                hits, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)
            ).sum())
    seconds = max(time.perf_counter() - started, 1e-9)
    result = {name: value / len(users) for name, value in totals.items()}
    result.update({"evaluated_users": len(users), "users_per_second": len(users) / seconds,
                   "scores_per_second": len(users) * data.n_items / seconds})
    return result


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logger_for(output / f"train_{run_id}.log")
    logger.info("EXPERIMENT_CONFIG %s", json.dumps(vars(args), sort_keys=True))
    raw, cache = load_data(args, logger)
    data = LLMRecData(raw, args.batch_size, args.feature_dim, args.seed)
    logger.info("dataset=%s users=%d items=%d train_interactions=%d cache=%s",
                args.dataset, data.n_users, data.n_items, data.n_train, cache)
    logger.info("AMAZON_MODALITY_ADAPTER text=hashed_ver4_metadata image=zeros llm_edges=train_proxy")

    ui = normalized_graph(data.train_matrix, device)
    iu = normalized_graph(data.train_matrix.T, device)
    graphs = (ui, iu, ui, iu, ui, iu)
    MM_Model, Decoder = import_upstream_model(args)
    model = MM_Model(data.n_users, data.n_items, args.embed_size, [64, 64], [0.1, 0.1],
                     data.image_features, data.text_features, data.user_features,
                     data.item_attributes).to(device)
    # Instantiated as upstream does; mask=False means it remains outside the loss.
    decoder = Decoder(data.user_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    _decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=2e-4)

    steps = data.n_train // args.batch_size + 1
    history, best_score, best_epoch, best_test, stale = [], -math.inf, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for _ in tqdm(range(steps), desc=f"LLMRec {epoch}/{args.epochs}"):
            users, positive, negative = data.sample()
            augmented_users = random.sample(
                users, int(len(users) * args.aug_sample_rate)
            )
            users = users + augmented_users
            positive = positive + [data.augmented_pairs[user][0] for user in augmented_users]
            negative = negative + [data.augmented_pairs[user][1] for user in augmented_users]
            outputs = model(*graphs)
            user_repr, item_repr = outputs[0], outputs[1]
            u = user_repr[users]; pos = item_repr[positive]; neg = item_repr[negative]
            mf, emb = bpr_loss(u, pos, neg, args.batch_size)
            image_mf, _ = bpr_loss(outputs[4][users], outputs[2][positive], outputs[2][negative], args.batch_size)
            text_mf, _ = bpr_loss(outputs[5][users], outputs[3][positive], outputs[3][negative], args.batch_size)
            augmented = 0.0
            for values in outputs[11].values():
                term, _ = bpr_loss(outputs[8][users], values[positive], values[negative], args.batch_size)
                augmented = augmented + term
            feature_reg = 0.5 * sum((tensor ** 2).sum() for tensor in outputs[2:6]) / data.n_items
            loss = mf + emb + 1e-5 * feature_reg + 0.012 * augmented + 0.0001 * (image_mf + text_mf)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach())
        valid = evaluate(model, graphs, data, "valid", args.batch_size * 2)
        score = valid["recall@10"]
        record = {"epoch": epoch, "loss": total / steps, "valid": valid}
        history.append(record); logger.info("epoch=%d loss=%.6f valid=%s", epoch, record["loss"], json.dumps(valid))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_test = evaluate(model, graphs, data, "test", args.batch_size * 2)
        else:
            stale += 1
            if stale > args.early_stopping_patience:
                logger.info("EARLY_STOP epoch=%d best_epoch=%d", epoch, best_epoch); break
    result = {"dataset": args.dataset, "best_epoch": best_epoch, "monitor_metric": "recall@10",
              "best_score": best_score, "test": best_test, "num_users": data.n_users,
              "evaluation_users": len(data.test_set), "num_items": data.n_items,
              "train_interactions": data.n_train, "visible_gpus": torch.cuda.device_count(),
              "history": history, "augmentation_source": "hashed_ver4_metadata_proxy"}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    logger.info("FINAL_RESULT %s", json.dumps(result)); logger.info("outputs=%s", output)


if __name__ == "__main__":
    main()
