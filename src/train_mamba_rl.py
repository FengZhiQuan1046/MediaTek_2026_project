"""Independent three-agent Mamba-RL training entry point."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.data_mamba_rl import load_recommendation_data
from src.model import MAMBA_MODEL_ID, load_or_encode_text
from src.model_mamba_rl import MultiAgentMambaRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Transition:
    user: int
    end: int
    target: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_agent_mamba_rl")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_transitions(data, maximum: int | None, seed: int) -> list[Transition]:
    """Build prefix-to-next-item transitions using bounded reservoir sampling."""
    rng, result, seen = random.Random(seed), [], 0
    for user, history in data.train_by_user.items():
        for end in range(1, len(history)):
            transition = Transition(user, end, history[end])
            seen += 1
            if maximum is None or len(result) < maximum:
                result.append(transition)
            else:
                replacement = rng.randrange(seen)
                if replacement < maximum:
                    result[replacement] = transition
    if not result:
        raise RuntimeError("No train prefixes exist; load more interactions or lower --min-user-events.")
    return result


def history_batch(data, transitions: list[Transition], max_history: int, device: str):
    histories = [data.train_by_user[row.user][max(0, row.end - max_history):row.end] for row in transitions]
    lengths = torch.tensor([len(history) for history in histories], device=device)
    padded = torch.zeros((len(histories), max(int(lengths.max()), 1)), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        padded[row, :len(history)] = torch.tensor(history, device=device)
    return padded, lengths, torch.tensor([row.target for row in transitions], device=device)


def evaluation_history_batch(data, users: list[int], split: str, max_history: int, device: str):
    histories = []
    for user in users:
        history = list(data.train_by_user[user])
        if split == "test":
            history.append(data.valid_target[user])
        histories.append(history[-max_history:])
    lengths = torch.tensor([len(history) for history in histories], device=device)
    padded = torch.zeros((len(users), max(int(lengths.max()), 1)), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        padded[row, :len(history)] = torch.tensor(history, device=device)
    return padded, lengths, histories


def candidate_slates(targets: torch.Tensor, num_items: int, count: int) -> torch.Tensor:
    negatives = torch.randint(num_items, (targets.size(0), count - 1), device=targets.device)
    while torch.any(negatives == targets.unsqueeze(1)):
        clashes = negatives == targets.unsqueeze(1)
        negatives[clashes] = torch.randint(num_items, (int(clashes.sum()),), device=targets.device)
    return torch.cat((targets.unsqueeze(1), negatives), dim=1)


def amp_context(device: str):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()


def train_stage(model, data, transitions, stage, epochs, batch_size, candidates, max_history,
                learning_rate, entropy_coef, supervised_coef, specialization_coef, device, logger):
    if epochs == 0:
        return []
    model.set_stage(stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    baselines = {"long": 0.0, "short": 0.0, "coordinator": 0.0}
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        random.shuffle(transitions)
        model.train()
        total, steps, started = 0.0, math.ceil(len(transitions) / batch_size), time.perf_counter()
        progress = tqdm(range(0, len(transitions), batch_size), total=steps,
                        desc=f"{stage} {epoch}/{epochs}", unit="step", dynamic_ncols=True)
        for start in progress:
            batch = transitions[start:start + batch_size]
            histories, lengths, targets = history_batch(data, batch, max_history, device)
            slates = candidate_slates(targets, data.num_items, candidates)
            with amp_context(device):
                output = model(histories, lengths, slates)
                labels = torch.zeros(len(batch), dtype=torch.long, device=device)
                reward_mean = 0.0
                if stage == "specialists":
                    loss = F.cross_entropy(output["long"], labels) + F.cross_entropy(output["short"], labels)
                elif stage == "coordinator":
                    loss = F.cross_entropy(output["coordinator"], labels)
                else:
                    distributions = {name: Categorical(logits=output[name])
                                     for name in ("long", "short", "coordinator")}
                    actions = {name: distribution.sample() for name, distribution in distributions.items()}
                    rewards = {name: torch.where(action == 0, torch.ones_like(action, dtype=torch.float),
                                                 torch.full_like(action, -0.05, dtype=torch.float))
                               for name, action in actions.items()}
                    policy_loss = 0.0
                    for name, distribution in distributions.items():
                        baselines[name] = 0.95 * baselines[name] + 0.05 * rewards[name].mean().item()
                        advantage = rewards[name] - baselines[name]
                        policy_loss -= (advantage.detach() * distribution.log_prob(actions[name])).mean()
                        policy_loss -= entropy_coef * distribution.entropy().mean()
                    long_state, short_state, _ = output["states"]
                    specialization = F.cosine_similarity(long_state, short_state).square().mean()
                    supervised = sum(F.cross_entropy(output[name], labels)
                                     for name in ("long", "short", "coordinator")) / 3
                    loss = policy_loss + supervised_coef * supervised + specialization_coef * specialization
                    reward_mean = rewards["coordinator"].mean().item()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", reward=f"{reward_mean:.3f}")
        average = total / steps
        epoch_losses.append(average)
        logger.info("stage=%s epoch=%d loss=%.6f transitions/s=%.2f",
                    stage, epoch, average, len(transitions) / max(time.perf_counter() - started, 1e-9))
    return epoch_losses


@torch.inference_mode()
def evaluate(model, data, split, batch_size, max_history, device, sample_count=0):
    model.eval()
    targets, users = (data.valid_target if split == "valid" else data.test_target), sorted(data.test_target)
    totals = {"recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    samples = []
    with amp_context(device):
        projected_items = model.project_all()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in tqdm(range(0, len(users), batch_size), desc=f"full-catalog {split}", unit="batch"):
        batch_users = users[start:start + batch_size]
        histories, lengths, raw_histories = evaluation_history_batch(data, batch_users, split, max_history, device)
        with amp_context(device):
            output = model.full_catalog_scores(histories, lengths, projected_items)
        scores = output["coordinator"].float()
        gold = torch.tensor([targets[user] for user in batch_users], device=device)
        for row, history in enumerate(raw_histories):
            seen = set(history) - {int(gold[row])}
            if seen:
                scores[row, list(seen)] = -torch.inf
        ranks = (scores >= scores.gather(1, gold.unsqueeze(1))).sum(1)
        for cutoff in (5, 10):
            totals[f"recall@{cutoff}"] += (ranks <= cutoff).sum().item()
            totals[f"ndcg@{cutoff}"] += torch.where(
                ranks <= cutoff, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)
            ).sum().item()
        remaining = max(sample_count - len(samples), 0)
        if remaining:
            top = torch.topk(scores, k=min(10, data.num_items), dim=1).indices
            for row in range(min(remaining, len(batch_users))):
                samples.append({
                    "user_index": batch_users[row], "history": raw_histories[row], "target": int(gold[row]),
                    "top_items": top[row].tolist(),
                    "agent_weights": {"long": round(float(output["weights"][row, 0]), 6),
                                      "short": round(float(output["weights"][row, 1]), 6)},
                })
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = max(time.perf_counter() - started, 1e-9)
    metrics = {name: value / len(users) for name, value in totals.items()}
    metrics.update({"users_per_second": len(users) / seconds,
                    "scores_per_second": len(users) * data.num_items / seconds})
    return metrics, samples


def generate_reasons(samples, data, cache_dir, device, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MAMBA_MODEL_ID, cache_dir=cache_dir)
    generator = AutoModelForCausalLM.from_pretrained(
        MAMBA_MODEL_ID, cache_dir=cache_dir,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()
    for sample in tqdm(samples, desc="Generating recommendation reasons", unit="user"):
        history = " | ".join(data.item_texts[item] for item in sample["history"][-10:])
        item = sample["top_items"][0]
        prompt = (
            "Explain this recommendation in one concise sentence using only the supplied history. "
            f"Long-term agent weight={sample['agent_weights']['long']}; "
            f"short-term agent weight={sample['agent_weights']['short']}. "
            f"History: {history}. Recommended item: {data.item_texts[item]}. Reason:"
        )
        encoded = tokenizer(prompt, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = generator.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
        reason = tokenizer.decode(output[0, encoded["input_ids"].size(1):], skip_special_tokens=True).strip()
        sample["recommendation"] = {
            "item_index": item, "item_text": data.item_texts[item],
            "reason": reason or "The item matches the user's recent and long-term interaction patterns.",
        }


def save_loss_curve(losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    offset = 0
    for stage, values in losses.items():
        xs = list(range(offset + 1, offset + len(values) + 1))
        axis.plot(xs, values, marker="o", label=stage)
        offset += len(values)
    axis.set(xlabel="Stage epoch", ylabel="Loss", title="Multi-agent Mamba-RL training")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Three-agent LoRA selective-Mamba recommender")
    parser.add_argument("--dataset", default="movielens-1m")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "cache"))
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=500_000)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-user-events", type=int, default=5)
    parser.add_argument("--specialist-epochs", type=int, default=1)
    parser.add_argument("--coordinator-epochs", type=int, default=1)
    parser.add_argument("--rl-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--short-window", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--specialist-lr", type=float, default=2e-4)
    parser.add_argument("--coordinator-lr", type=float, default=2e-4)
    parser.add_argument("--joint-lr", type=float, default=5e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--supervised-coef", type=float, default=0.1)
    parser.add_argument("--specialization-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-mamba", action="store_true")
    parser.add_argument("--generate-reasons", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reason-count", type=int, default=20)
    parser.add_argument("--reason-max-new-tokens", type=int, default=40)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs_mamba_rl"))
    args = parser.parse_args()
    if min(args.specialist_epochs, args.coordinator_epochs, args.rl_epochs) < 0:
        parser.error("stage epoch counts cannot be negative")
    if args.candidates < 2 or args.batch_size < 1 or args.max_history < 1:
        parser.error("candidate count must be >=2 and batch/history sizes must be positive")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = args.dataset.replace(":", "_").replace("/", "_")
    output = Path(args.output_dir) / safe_dataset / run_id
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output / "train.log")
    logger.info("run_id=%s dataset=%s device=%s", run_id, args.dataset, args.device)
    data = load_recommendation_data(
        args.dataset, args.data_path, args.cache_dir, args.max_events, args.min_rating, args.min_user_events
    )
    logger.info("users=%d items=%d train_interactions=%d",
                data.num_users, data.num_items, sum(map(len, data.train_by_user.values())))

    fingerprint = hashlib.sha1("\n".join(data.item_texts).encode("utf-8")).hexdigest()[:12]
    artifact = Path(args.cache_dir) / "mamba_multi_agent" / f"{safe_dataset}_{data.num_items}_{fingerprint}.pt"
    if args.skip_mamba:
        generator = torch.Generator().manual_seed(args.seed)
        item_features = torch.randn(data.num_items, args.dim, generator=generator)
        logger.warning("--skip-mamba uses random item features; it is only a smoke-test mode")
    else:
        item_features = load_or_encode_text(data.item_texts, str(artifact), args.device, False, args.cache_dir)
        assert item_features is not None
    item_features = item_features.to(dtype=torch.float16 if args.device.startswith("cuda") else torch.float32)
    model = MultiAgentMambaRecommender(
        item_features, args.dim, args.lora_rank, args.lora_alpha, args.lora_dropout, args.short_window
    ).to(args.device)
    logger.info("agent_parameters=%s", model.agent_parameter_counts())
    transitions = build_transitions(data, args.max_transitions, args.seed)
    logger.info("training_transitions=%d", len(transitions))
    common = dict(
        model=model, data=data, transitions=transitions, batch_size=args.batch_size,
        candidates=args.candidates, max_history=args.max_history, entropy_coef=args.entropy_coef,
        supervised_coef=args.supervised_coef, specialization_coef=args.specialization_coef,
        device=args.device, logger=logger,
    )
    losses = {
        "specialists": train_stage(stage="specialists", epochs=args.specialist_epochs,
                                   learning_rate=args.specialist_lr, **common),
        "coordinator": train_stage(stage="coordinator", epochs=args.coordinator_epochs,
                                   learning_rate=args.coordinator_lr, **common),
        "joint": train_stage(stage="joint", epochs=args.rl_epochs, learning_rate=args.joint_lr, **common),
    }
    valid_metrics, _ = evaluate(model, data, "valid", args.eval_batch_size, args.max_history, args.device)
    test_metrics, samples = evaluate(
        model, data, "test", args.eval_batch_size, args.max_history, args.device, args.reason_count
    )
    logger.info("VALID %s", json.dumps(valid_metrics, sort_keys=True))
    logger.info("TEST %s", json.dumps(test_metrics, sort_keys=True))
    checkpoint = {
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "config": vars(args), "valid_metrics": valid_metrics, "test_metrics": test_metrics,
        "item_vector_artifact": str(artifact),
    }
    torch.save(checkpoint, output / "multi_agent_lora.pt")
    save_loss_curve(losses, output / "loss.png")
    if args.generate_reasons and samples:
        model.to("cpu")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        generate_reasons(samples, data, args.cache_dir, args.device, args.reason_max_new_tokens)
    else:
        for sample in samples:
            item = sample["top_items"][0]
            sample["recommendation"] = {"item_index": item, "item_text": data.item_texts[item], "reason": None}
    (output / "recommendations.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(
        json.dumps({"valid": valid_metrics, "test": test_metrics}, indent=2), encoding="utf-8"
    )
    logger.info("outputs=%s", output)


if __name__ == "__main__":
    main()

