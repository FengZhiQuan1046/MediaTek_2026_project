"""Train LLaRA on ver4 data and evaluate against the entire item catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping  # noqa: E402
from pytorch_lightning.strategies import DDPStrategy  # noqa: E402
from local_model import MInterface  # noqa: E402
from sasrec import SASRec  # noqa: E402
from amazon_data import AmazonData, TrainCollater  # noqa: E402
from data_adapter import load_ver4_data  # noqa: E402
from prepare_rec import train_rec_model  # noqa: E402
from ranking import full_catalog_scores, rank_like_ver4, ranking_metrics  # noqa: E402

VER4_ROOT = ROOT.parents[1] / "ver4"
if str(VER4_ROOT) not in sys.path:
    sys.path.insert(0, str(VER4_ROOT))
from src.train_mamba_rl import build_catalog_priors  # noqa: E402
from src.sequence_length_metrics import SequenceLengthMetrics  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--llm-path", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--eval-user-limit", type=int, default=0)
    parser.add_argument("--catalog-chunk-size", type=int, default=8)
    parser.add_argument("--popularity-alpha", type=float, default=None)
    parser.add_argument("--transition-beta", type=float, default=None)
    parser.add_argument("--rec-epochs", type=int, default=10)
    parser.add_argument("--rec-batch-size", type=int, default=256)
    parser.add_argument("--rec-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=25252)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    args = parser.parse_args()
    # Match the subset settings in ver4/run_long_short.sh for direct CLI runs.
    subset_protocols = {
        "amazon-all-beauty": (500000, -0.25, 4.0),
        "amazon:Baby_Products": (1500000, 0.30, 0.5),
        "amazon-sports-and-outdoors": (1000000, 0.35, 0.5),
        "amazon-toys-and-games": (1500000, 0.30, 0.5),
        "amazon-books": (2000000, 0.35, 0.5),
        "amazon-video-games": (1000000, 0.20, 0.5),
        "amazon-clothing-shoes-and-jewelry": (2000000, 0.35, 0.5),
        "amazon:Beauty_and_Personal_Care": (2000000, 0.35, 0.5),
    }
    maximum, alpha, beta = subset_protocols.get(args.dataset, (500000, 0.0, 0.0))
    if args.max_train_samples is None:
        args.max_train_samples = maximum
    if args.popularity_alpha is None:
        args.popularity_alpha = alpha
    if args.transition_beta is None:
        args.transition_beta = beta
    if min(args.devices, args.batch_size, args.accumulate_grad_batches, args.max_epochs,
           args.maxlen, args.catalog_chunk_size, args.rec_epochs, args.rec_batch_size,
           args.rec_size) < 1:
        parser.error("device count, epochs, model sizes and batch sizes must be positive")
    if args.max_train_samples < 0 or args.eval_user_limit < 0 or args.early_stopping_patience < 0:
        parser.error("sample limits and stopping patience must be non-negative")
    if args.lr <= 0 or args.num_workers < 0:
        parser.error("learning rate must be positive and num_workers non-negative")
    return args


class ReproductionInterface(MInterface):
    """LLaRA training with ver4's exact full-catalog evaluation protocol."""
    def wrap_emb(self, batch):
        ids = batch["tokens"].input_ids
        embeddings = self.llama_model.get_input_embeddings()(ids).clone()
        values = self.encode_items(batch["seq"])
        token_id = self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",
                                        add_special_tokens=False).input_ids.item()
        for row in range(len(ids)):
            positions = (ids[row] == token_id).nonzero(as_tuple=True)[0]
            count = int(batch["len_seq"][row])
            if batch["flag"]:
                if len(positions):
                    raise ValueError("Placeholders must be hidden during text-only training")
            elif len(positions) != count:
                raise ValueError("History marker count differs from input items")
            if len(positions):
                embeddings[row, positions] = values[row, :count]
        return embeddings

    def load_rec_model(self, rec_model_path):
        checkpoint = torch.load(rec_model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError("Unsupported SASRec cache format; remove cache and retrain")
        self.rec_model = SASRec(checkpoint["hidden_size"], checkpoint["item_num"],
                                checkpoint["state_size"])
        self.rec_model.load_state_dict(checkpoint["state_dict"])
        self.rec_model.eval()
        for parameter in self.rec_model.parameters():
            parameter.requires_grad_(False)

    def configure_evaluation(self, data, names, chunk_size, popularity_alpha, transition_beta):
        self.eval_data = data
        self.eval_chunk_size = chunk_size
        self.popularity_alpha = popularity_alpha
        self.transition_beta = transition_beta
        encoded = self.llama_tokenizer([name + "\n" for name in names],
                                       add_special_tokens=False, padding=True,
                                       return_tensors="pt")
        self.catalog_ids = encoded.input_ids
        self.catalog_masks = encoded.attention_mask
        self.catalog_priors = build_catalog_priors(data, "cpu") if (
            popularity_alpha != 0.0 or transition_beta != 0.0) else None

    def on_fit_start(self):
        self.rec_model.device = str(self.device)

    def on_validation_start(self):
        self.rec_model.device = str(self.device)
        self.catalog_ids = self.catalog_ids.to(self.device, non_blocking=True)
        self.catalog_masks = self.catalog_masks.to(self.device, non_blocking=True)

    def on_test_start(self):
        self.rec_model.device = str(self.device)
        self.catalog_ids = self.catalog_ids.to(self.device, non_blocking=True)
        self.catalog_masks = self.catalog_masks.to(self.device, non_blocking=True)

    def _start_evaluation(self, split):
        self.eval_ranks = {}
        self.eval_split = split
        self.eval_started = self.eval_last_report = time.perf_counter()
        self.eval_completed = 0
        users = len(self.eval_data.valid_target if split == "valid" else self.eval_data.test_target)
        if self.eval_user_limit > 0:
            users = min(users, self.eval_user_limit)
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.eval_local_users = math.ceil(users / world)
        if self.trainer.is_global_zero:
            print(f"EVAL_START split={split} epoch={self.current_epoch + 1} "
                  f"users={users} catalog_items={self.catalog_ids.size(0)} "
                  f"chunk_size={self.eval_chunk_size}", flush=True)

    def on_validation_epoch_start(self):
        self._start_evaluation("valid")

    def on_test_epoch_start(self):
        self._start_evaluation("test")

    def _eval_progress(self, items_done, items_total):
        if not self.trainer.is_global_zero:
            return
        now = time.perf_counter()
        if now - self.eval_last_report < 30:
            return
        elapsed = now - self.eval_started
        completed_items = self.eval_completed * items_total + items_done
        target_items = self.eval_local_users * items_total
        eta = (target_items - completed_items) * elapsed / max(completed_items, 1)
        print(f"EVAL_PROGRESS split={self.eval_split} epoch={self.current_epoch + 1} "
              f"users={self.eval_completed}/{self.eval_local_users} "
              f"current_user_items={items_done}/{items_total} "
              f"elapsed_min={elapsed / 60:.1f} eta_min={eta / 60:.1f}", flush=True)
        self.eval_last_report = now

    @torch.no_grad()
    def _evaluate_batch(self, batch, split):
        embeddings = self.wrap_emb(batch)
        for row, user in enumerate(batch["user"]):
            length = int(batch["tokens"].attention_mask[row].sum())
            scores = full_catalog_scores(self.llama_model, self.catalog_ids,
                                         self.catalog_masks, embeddings[row, :length],
                                         self.eval_chunk_size,
                                         progress=self._eval_progress if self.trainer.is_global_zero else None)
            # Use raw chronological history for the prior and seen-item mask,
            # matching ver4 even when a model requests reversed input order.
            history = list(self.eval_data.train_by_user[user])
            if split == "test":
                history.append(self.eval_data.valid_target[user])
            history = history[-self.eval_data_maxlen:]
            target = int(batch["item_id"][row])
            rank = rank_like_ver4(scores, history, target, self.catalog_priors,
                                  self.popularity_alpha, self.transition_beta)
            self.eval_ranks.setdefault(user, rank)
            self.eval_completed += 1
        return None

    def validation_step(self, batch, batch_idx):
        return self._evaluate_batch(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self._evaluate_batch(batch, "test")

    def _end_evaluation(self, prefix):
        parts = [self.eval_ranks]
        if dist.is_available() and dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, self.eval_ranks)
        merged = {}
        for part in parts:
            for user, rank in part.items():
                merged.setdefault(user, rank)
        expected = len(self.eval_data.valid_target if prefix == "val" else self.eval_data.test_target)
        if self.eval_user_limit > 0:
            expected = min(expected, self.eval_user_limit)
        if len(merged) != expected:
            raise RuntimeError(f"{prefix}: evaluated {len(merged)} users, expected {expected}")
        self.final_eval_ranks = merged
        if self.trainer.is_global_zero:
            print(f"EVAL_DONE split={self.eval_split} epoch={self.current_epoch + 1} "
                  f"users={len(merged)} elapsed_min={(time.perf_counter() - self.eval_started) / 60:.1f}",
                  flush=True)
        for name, value in ranking_metrics(merged.values()).items():
            self.log(f"{prefix}_{name}", value, on_epoch=True, prog_bar=name == "ndcg@10",
                     sync_dist=True)

    def on_validation_epoch_end(self):
        self._end_evaluation("val")

    def on_test_epoch_end(self):
        self._end_evaluation("test")


class BestAdapter(pl.Callback):
    """Keep the best validation LoRA and projector without copying the frozen 7B base."""
    def __init__(self, output):
        self.path = Path(output) / "best_adapter.pt"
        self.best = -math.inf
        self.epoch = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        score = float(trainer.callback_metrics["val_ndcg@10"])
        if score > self.best:
            self.best = score
            self.epoch = trainer.current_epoch
            if trainer.is_global_zero:
                state = {name: parameter.detach().cpu().clone()
                         for name, parameter in pl_module.named_parameters()
                         if "lora_" in name or name.startswith("projector.")}
                torch.save({"state_dict": state, "epoch": self.epoch,
                            "val_ndcg@10": score}, self.path)
                print(f"BEST_ADAPTER epoch={self.epoch + 1} val_ndcg@10={score:.6f} "
                      f"path={self.path}", flush=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def restore(self, model):
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(payload["state_dict"], strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys}")
        return payload


class AmazonModule(pl.LightningDataModule):
    def __init__(self, args, data, tokenizer, prompts):
        super().__init__()
        self.args, self.data = args, data
        self.tokenizer, self.prompts = tokenizer, prompts
        self.trainset = AmazonData(data, "train", args.maxlen,
                                   args.max_train_samples, args.eval_user_limit, args.seed)
        self.valset = AmazonData(data, "valid", args.maxlen,
                                 args.max_train_samples, args.eval_user_limit, args.seed)
        self.testset = AmazonData(data, "test", args.maxlen,
                                  args.max_train_samples, args.eval_user_limit, args.seed)

    def train_dataloader(self):
        steps = self.args.max_epochs * max(len(self.trainset) // self.args.batch_size, 1)
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True,
                          drop_last=False, num_workers=self.args.num_workers,
                          pin_memory=True, persistent_workers=self.args.num_workers > 0,
                          collate_fn=TrainCollater(self.tokenizer, self.prompts, True, steps))

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers,
                          pin_memory=True, persistent_workers=self.args.num_workers > 0,
                          collate_fn=TrainCollater(self.tokenizer, self.prompts, False))

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers,
                          pin_memory=True, persistent_workers=self.args.num_workers > 0,
                          collate_fn=TrainCollater(self.tokenizer, self.prompts, False))


def scalar(value):
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def main():
    args = arguments()
    from transformers import AutoConfig
    try:
        AutoConfig.from_pretrained(args.llm_path, cache_dir=args.cache_dir)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot access Llama model {args.llm_path!r}.") from error
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, interaction_cache = load_ver4_data(args.dataset, args.cache_dir,
                                             args.max_events, args.min_rating)
    safe = args.dataset.replace(":", "_").replace("/", "_")
    identity = {"rec_cache_version": 1, "interaction_cache": str(interaction_cache),
                "maxlen": args.maxlen, "rec_epochs": args.rec_epochs,
                "rec_size": args.rec_size, "rec_batch_size": args.rec_batch_size,
                "seed": args.seed}
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    rec_path = Path(args.cache_dir) / "llara" / f"{safe}_{digest}_sasrec.pt"
    train_rec_model(data, rec_path, "cuda", args.rec_epochs, args.rec_batch_size,
                    args.rec_size, args.maxlen, seed=args.seed)
    hparams = dict(llm_path=args.llm_path, rec_model_path=str(rec_path),
                   model_name="mlp_projector", rec_size=args.rec_size, loss="lm",
                   llm_tuning="lora", peft_dir=None, peft_config=None,
                   lora_r=8, lora_alpha=32, lora_dropout=0.1,
                   rec_embed="SASRec", output_dir=str(output), save="part",
                   lr=args.lr, weight_decay=1e-5, lr_scheduler="cosine",
                   lr_decay_min_lr=8e-6, lr_warmup_start_lr=8e-6)
    model = ReproductionInterface(**hparams)
    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.enable_input_require_grads()
    prompts = [line.strip() for line in (ROOT / "prompt.txt").read_text().splitlines() if line.strip()]
    module = AmazonModule(args, data, model.llama_tokenizer, prompts)
    print(f"EVAL_PLAN validation_users={len(module.valset)} test_users={len(module.testset)} "
          f"catalog_items={data.num_items} chunk_size={args.catalog_chunk_size}", flush=True)
    model.configure_evaluation(data, module.trainset.names, args.catalog_chunk_size,
                               args.popularity_alpha, args.transition_beta)
    model.eval_data_maxlen = args.maxlen
    model.eval_user_limit = args.eval_user_limit
    best = BestAdapter(output)
    callbacks = [best, EarlyStopping(monitor="val_ndcg@10", mode="max",
                                     patience=args.early_stopping_patience, min_delta=0.001)]
    strategy = DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto"
    samples_per_rank = math.ceil(len(module.trainset) / args.devices)
    batches_per_rank = math.ceil(samples_per_rank / args.batch_size)
    if batches_per_rank < 1:
        raise ValueError("Training examples are fewer than devices * batch_size")
    max_steps = args.max_epochs * math.ceil(batches_per_rank / args.accumulate_grad_batches)
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=strategy,
                         precision="bf16-mixed", max_epochs=args.max_epochs,
                         max_steps=max_steps,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         callbacks=callbacks, logger=False, enable_checkpointing=False,
                         check_val_every_n_epoch=1, num_sanity_val_steps=0)
    trainer.fit(model=model, datamodule=module)
    best_payload = best.restore(model)
    ranked_validation = trainer.validate(model=model, datamodule=module, verbose=False)[0]
    validation_ranks = model.final_eval_ranks
    test_results = trainer.test(model=model, datamodule=module, verbose=False)[0]
    test_ranks = model.final_eval_ranks
    metric_names = ("ndcg@5", "ndcg@10", "recall@5", "recall@10", "hit@5", "hit@10")
    validation = {name: scalar(ranked_validation["val_" + name]) for name in metric_names}
    test = {name: scalar(test_results["test_" + name]) for name in metric_names}
    for split, report, ranks in (("valid", validation, validation_ranks),
                                  ("test", test, test_ranks)):
        targets = data.valid_target if split == "valid" else data.test_target
        lengths = {user: len(data.train_by_user[user]) + int(user in data.valid_target)
                   + int(user in data.test_target) for user in targets}
        groups = SequenceLengthMetrics(data.train_by_user, targets, 10,
                                       lengths=lengths, basis="filtered_full_sequence")
        groups.update(list(ranks), list(ranks.values()))
        report["evaluated_users"] = len(ranks)
        report["total_users"] = len(targets)
        report["sequence_length"] = groups.report()
    if trainer.is_global_zero:
        result = {"method": "LLaRA", "dataset": args.dataset,
                  "validation": validation, "test": test,
                  "num_users": data.num_users, "num_items": data.num_items,
                  "train_interactions": sum(map(len, data.train_by_user.values())),
                  "train_examples": len(module.trainset),
                  "validation_examples": len(module.valset),
                  "test_examples": len(module.testset),
                  "visible_gpus": torch.cuda.device_count(),
                  "ranking_protocol": "ver4_full_catalog",
                  "ranking_score": "mean_token_log_probability_of_item_name",
                  "evaluation_scope": "entire_catalog_all_eligible_users" if args.eval_user_limit == 0 else "pilot_limited_users",
                  "seen_item_mask": "ver4_seen_minus_target",
                  "tie_rule": "scores_greater_or_equal_gold",
                  "popularity_alpha": args.popularity_alpha,
                  "transition_beta": args.transition_beta,
                  "max_history": args.maxlen,
                  "best_epoch": best_payload["epoch"],
                  "interaction_cache": str(interaction_cache),
                  "rec_model_cache": str(rec_path)}
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print("FINAL_RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
