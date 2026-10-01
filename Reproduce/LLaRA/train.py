"""Run upstream LLaRA on sessions produced from exact ver4 data."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("LLARA_UPSTREAM_DIR", ROOT.parents[2] / "LLaRA")).expanduser().resolve()
if not (UPSTREAM / "model" / "model_interface.py").is_file():
    raise RuntimeError(
        f"LLaRA model source not found at {UPSTREAM}; "
        "set LLARA_UPSTREAM_DIR to a complete local LLaRA checkout"
    )
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping  # noqa: E402
from pytorch_lightning.strategies import DDPStrategy  # noqa: E402
from model.model_interface import MInterface  # noqa: E402
from amazon_data import AmazonData, TrainCollater  # noqa: E402
from data_adapter import load_ver4_data  # noqa: E402
from prepare_rec import train_rec_model  # noqa: E402
from ranking import candidate_scores, ranking_metrics  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--max-events", type=int)
    parser.add_argument("--min-rating", type=float, default=4.0); parser.add_argument("--llm-path", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--devices", type=int, default=2); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--accumulate-grad-batches", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=5); parser.add_argument("--maxlen", type=int, default=10)
    parser.add_argument("--cans-num", type=int, default=10); parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--eval-user-limit", type=int, default=1000); parser.add_argument("--rec-epochs", type=int, default=10)
    parser.add_argument("--rec-batch-size", type=int, default=256); parser.add_argument("--rec-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4); parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    args = parser.parse_args()
    if min(args.devices, args.batch_size, args.accumulate_grad_batches, args.max_epochs,
           args.maxlen, args.cans_num, args.rec_epochs, args.rec_batch_size,
           args.rec_size) < 1:
        parser.error("device count, epochs, model sizes and batch sizes must be positive")
    if args.max_train_samples < 0 or args.eval_user_limit < 0 or args.early_stopping_patience < 0:
        parser.error("sample limits and stopping patience must be non-negative")
    if args.lr <= 0 or args.num_workers < 0:
        parser.error("learning rate must be positive and num_workers non-negative")
    return args


class ReproductionInterface(MInterface):
    """Aggregate candidate evaluation across Lightning ranks and move SASRec indices."""
    def wrap_emb(self, batch):
        # Upstream writes into an embedding leaf view; clone keeps autograd valid.
        ids = batch["tokens"].input_ids
        embeddings = self.llama_model.get_input_embeddings()(ids).clone()
        replacements = (
            ("[HistoryEmb]", self.encode_items(batch["seq"]), batch["len_seq"]),
            ("[CansEmb]", self.encode_items(batch["cans"]), batch["len_cans"]),
            ("[ItemEmb]", self.encode_items(batch["item_id"]), None),
        )
        for marker, values, lengths in replacements:
            token_id = self.llama_tokenizer(marker, return_tensors="pt",
                                           add_special_tokens=False).input_ids.item()
            for row in range(len(ids)):
                positions = (ids[row] == token_id).nonzero(as_tuple=True)[0]
                if len(positions):
                    count = int(lengths[row]) if lengths is not None else 1
                    if len(positions) != count:
                        raise ValueError(f"{marker} count differs from input items")
                    selected = values[row, :count] if lengths is not None else values[row].unsqueeze(0)
                    embeddings[row, positions] = selected
        return embeddings

    def load_rec_model(self, rec_model_path):
        # This is the trusted, locally generated upstream SASRec object.
        self.rec_model = torch.load(rec_model_path, map_location="cpu", weights_only=False)
        self.rec_model.eval()
        for parameter in self.rec_model.parameters():
            parameter.requires_grad_(False)

    ranking_enabled = False

    @torch.no_grad()
    def _rank_batch(self, batch):
        embeddings = self.wrap_emb(batch)
        ranks = []
        for row, names in enumerate(batch["cans_name"]):
            prompt_length = int(batch["tokens"].attention_mask[row].sum())
            scores = candidate_scores(
                self.llama_model, self.llama_tokenizer,
                embeddings[row, :prompt_length], names,
            ).tolist()
            candidates = batch["cans"][row].tolist()
            target = int(batch["item_id"][row])
            order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
            ranks.append(next(rank for rank, index in enumerate(order, 1)
                              if candidates[index] == target))
        return ranks

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        results = super().validation_step(batch, batch_idx)
        if self.ranking_enabled:
            return [(*result, rank) for result, rank in zip(results, self._rank_batch(batch))]
        return results

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        results = super().test_step(batch, batch_idx)
        if self.ranking_enabled:
            return [(*result, rank) for result, rank in zip(results, self._rank_batch(batch))]
        return results

    def on_fit_start(self):
        self.rec_model.device = str(self.device)

    def on_validation_start(self):
        self.rec_model.device = str(self.device)

    def on_test_start(self):
        self.rec_model.device = str(self.device)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        for user, result in zip(batch["user"], outputs):
            generate, real, cans = result[:3]
            self.val_content["user"].append(user)
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)
            if self.ranking_enabled:
                self.val_ranks.setdefault(user, result[3])

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        for user, result in zip(batch["user"], outputs):
            generate, real, cans = result[:3]
            self.test_content["user"].append(user)
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)
            if self.ranking_enabled:
                self.test_ranks.setdefault(user, result[3])

    def on_validation_epoch_start(self):
        self.val_content = {key: [] for key in ("user", "generate", "real", "cans")}
        self.val_ranks = {}

    def on_test_epoch_start(self):
        self.test_content = {key: [] for key in ("user", "generate", "real", "cans")}
        self.test_ranks = {}

    @staticmethod
    def _merge_content(content):
        parts = [content]
        if dist.is_available() and dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, content)
        # Lightning's distributed sampler can pad the final batch with duplicate users.
        rows = {}
        for part in parts:
            for user, generate, real, cans in zip(*(part[key] for key in
                                                    ("user", "generate", "real", "cans"))):
                rows.setdefault(user, (generate, real, cans))
        return {key: [row[index] for row in rows.values()]
                for index, key in enumerate(("generate", "real", "cans"))}

    @staticmethod
    def _merge_ranks(ranks):
        parts = [ranks]
        if dist.is_available() and dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, ranks)
        merged = {}
        for part in parts:
            for user, rank in part.items():
                merged.setdefault(user, rank)
        return merged

    def on_validation_epoch_end(self):
        content = self._merge_content(self.val_content)
        ratio, hr = self.calculate_hr1(content)
        self.log("val_prediction_valid", ratio, on_epoch=True, sync_dist=True)
        self.log("val_hr", hr, on_epoch=True, sync_dist=True)
        self.log("metric", ratio * hr, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.ranking_enabled:
            for name, value in ranking_metrics(self._merge_ranks(self.val_ranks).values()).items():
                self.log("val_" + name, value, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        content = self._merge_content(self.test_content)
        ratio, hr = self.calculate_hr1(content)
        self.log("test_prediction_valid", ratio, on_epoch=True, sync_dist=True)
        self.log("test_hr", hr, on_epoch=True, sync_dist=True)
        self.log("metric", ratio * hr, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.ranking_enabled:
            for name, value in ranking_metrics(self._merge_ranks(self.test_ranks).values()).items():
                self.log("test_" + name, value, on_epoch=True, sync_dist=True)


class AmazonModule(pl.LightningDataModule):
    def __init__(self, args, data, tokenizer, prompts):
        super().__init__(); self.args, self.data = args, data
        self.tokenizer, self.prompts = tokenizer, prompts
        self.trainset = AmazonData(data, "train", args.cans_num, args.maxlen,
                                   args.max_train_samples, args.eval_user_limit, args.seed)
        self.valset = AmazonData(data, "valid", args.cans_num, args.maxlen,
                                 args.max_train_samples, args.eval_user_limit, args.seed)
        self.testset = AmazonData(data, "test", args.cans_num, args.maxlen,
                                  args.max_train_samples, args.eval_user_limit, args.seed)

    def train_dataloader(self):
        steps = self.args.max_epochs * max(len(self.trainset) // self.args.batch_size, 1)
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True,
                          drop_last=True, num_workers=self.args.num_workers,
                          collate_fn=TrainCollater(self.tokenizer, self.prompts, True, steps))

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers,
                          collate_fn=TrainCollater(self.tokenizer, self.prompts, False))

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers,
                          collate_fn=TrainCollater(self.tokenizer, self.prompts, False))


def scalar(value):
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def main():
    args = arguments()
    # Check gated model access before the expensive SASRec pretraining stage.
    from transformers import AutoConfig
    try:
        AutoConfig.from_pretrained(args.llm_path, cache_dir=args.cache_dir)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot access Llama model {args.llm_path!r}. Set LLM_PATH to a "
            "readable local snapshot, or authenticate for the gated model."
        ) from error
    pl.seed_everything(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    data, interaction_cache = load_ver4_data(args.dataset, args.cache_dir, args.max_events, args.min_rating)
    safe = args.dataset.replace(":", "_").replace("/", "_")
    identity = {"interaction_cache": str(interaction_cache), "maxlen": args.maxlen,
                "rec_epochs": args.rec_epochs, "rec_size": args.rec_size,
                "rec_batch_size": args.rec_batch_size, "seed": args.seed}
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    rec_path = Path(args.cache_dir) / "llara" / f"{safe}_{digest}_sasrec.pt"
    train_rec_model(data, rec_path, "cuda", args.rec_epochs, args.rec_batch_size,
                    args.rec_size, args.maxlen, seed=args.seed)
    prompt_path = ROOT / "prompt.txt"
    hparams = dict(
        llm_path=args.llm_path, rec_model_path=str(rec_path), model_name="mlp_projector",
        rec_size=args.rec_size, loss="lm", llm_tuning="lora", peft_dir=None,
        peft_config=None, lora_r=8, lora_alpha=32, lora_dropout=0.1,
        rec_embed="SASRec", output_dir=str(output), save="part",
        lr=args.lr, weight_decay=1e-5, lr_scheduler="cosine",
        lr_decay_min_lr=8e-6, lr_warmup_start_lr=8e-6,
    )
    model = ReproductionInterface(**hparams)
    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.enable_input_require_grads()
    prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]
    module = AmazonModule(args, data, model.llama_tokenizer, prompts)
    callbacks = [EarlyStopping(monitor="metric", mode="max", patience=args.early_stopping_patience, min_delta=0.001)]
    strategy = DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto"
    samples_per_rank = math.ceil(len(module.trainset) / args.devices)
    batches_per_rank = samples_per_rank // args.batch_size
    if batches_per_rank < 1:
        raise ValueError("Training examples are fewer than devices * batch_size")
    max_steps = args.max_epochs * math.ceil(batches_per_rank / args.accumulate_grad_batches)
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, strategy=strategy, precision="bf16-mixed",
        max_epochs=args.max_epochs, max_steps=max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks, logger=False, enable_checkpointing=False,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model=model, datamodule=module)
    model.ranking_enabled = True
    ranked_validation = trainer.validate(model=model, datamodule=module, verbose=False)[0]
    metric_names = ("ndcg@5", "ndcg@10", "recall@5", "recall@10")
    validation = {str(key): scalar(value) for key, value in ranked_validation.items()
                  if str(key).startswith("val_") or str(key) == "metric"}
    validation.update({name: validation["val_" + name] for name in metric_names})
    test_results = trainer.test(model=model, datamodule=module, verbose=False)[0]
    test = {str(k): scalar(v) for k, v in test_results.items()
            if str(k).startswith("test_") or str(k) == "metric"}
    test.update({name: test["test_" + name] for name in metric_names})
    if trainer.is_global_zero:
        result = {"method": "LLaRA", "dataset": args.dataset, "validation": validation, "test": test,
                  "num_users": data.num_users, "num_items": data.num_items,
                  "train_interactions": sum(map(len, data.train_by_user.values())),
                  "train_examples": len(module.trainset), "validation_examples": len(module.valset),
                  "test_examples": len(module.testset), "visible_gpus": torch.cuda.device_count(),
                  "ranking_protocol": "sampled_candidates",
                  "ranking_score": "mean_token_log_probability_of_candidate_name",
                  "candidate_count": args.cans_num,
                  "interaction_cache": str(interaction_cache), "rec_model_cache": str(rec_path)}
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print("FINAL_RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
