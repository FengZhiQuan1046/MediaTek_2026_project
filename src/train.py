from __future__ import annotations

import argparse
import random

import torch
from torch.nn import functional as F

from src.data import build_data, edge_index, load_amazon, synthetic_events
from src.model import HybridRecommender, load_or_encode_text


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
    p.add_argument("--artifact", default="artifacts/item_text_embeddings.pt")
    p.add_argument("--skip-mamba", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    random.seed(42); torch.manual_seed(42)

    events = synthetic_events() if args.synthetic else load_amazon(args.subset, args.max_events)
    data = build_data(events)
    print(f"users={data.num_users}, items={data.num_items}, train_edges={sum(map(len, data.train_by_user.values()))}")
    text = load_or_encode_text(data.item_texts, args.artifact, args.device, args.skip_mamba)
    model = HybridRecommender(data.num_users, data.num_items, args.dim, text).to(args.device)
    edges = edge_index(data).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    train_users = list(data.train_by_user)

    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_users); losses = []
        for start in range(0, len(train_users), args.batch_size):
            users = torch.tensor(train_users[start:start + args.batch_size], device=args.device)
            positive = torch.tensor([random.choice(data.train_by_user[int(u)]) for u in users.tolist()], device=args.device)
            negative = torch.randint(data.num_items, positive.shape, device=args.device)
            # BPR needs a genuinely unobserved comparison item for each draw.
            while torch.any(negative == positive):
                clash = negative == positive
                negative[clash] = torch.randint(data.num_items, (int(clash.sum()),), device=args.device)
            candidates = torch.stack((positive, negative), 1)
            scores = model.score(users, candidates, edges, padded_histories(data, users, args.device))
            loss = F.softplus(-(scores[:, 0] - scores[:, 1])).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        recall, ndcg = evaluate(model, data, edges, data.valid_target, args.device)
        print(f"epoch={epoch} loss={sum(losses)/len(losses):.4f} valid Recall@10={recall:.4f} NDCG@10={ndcg:.4f}")
    recall, ndcg = evaluate(model, data, edges, data.test_target, args.device)
    print(f"TEST Recall@10={recall:.4f} NDCG@10={ndcg:.4f}")


if __name__ == "__main__":
    main()
