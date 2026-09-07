"""Run upstream LLaRA on sessions produced from exact ver4 data."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parents[2] / "LLaRA"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping  # noqa: E402
from pytorch_lightning.strategies import DDPStrategy  # noqa: E402
from model.model_interface import MInterface  # noqa: E402
from amazon_data import AmazonData, TrainCollater  # noqa: E402
from data_adapter import load_ver4_data  # noqa: E402
from prepare_rec import train_rec_model  # noqa: E402


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
    return parser.parse_args()


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
    args = arguments(); pl.seed_everything(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    data, interaction_cache = load_ver4_data(args.dataset, args.cache_dir, args.max_events, args.min_rating)
    safe = args.dataset.replace(":", "_").replace("/", "_")
    rec_path = Path(args.cache_dir) / "llara" / f"{safe}_sasrec.pt"
    train_rec_model(data, rec_path, "cuda", args.rec_epochs, args.rec_batch_size,
                    args.rec_size, args.maxlen, seed=args.seed)
    artifact_output = Path(args.cache_dir) / "llara" / "generation" / safe
    artifact_output.mkdir(parents=True, exist_ok=True)
    prompt_path = ROOT / "prompt.txt"
    hparams = dict(
        llm_path=args.llm_path, rec_model_path=str(rec_path), model_name="mlp_projector",
        rec_size=args.rec_size, loss="lm", llm_tuning="lora", peft_dir=None,
        peft_config=None, lora_r=8, lora_alpha=32, lora_dropout=0.1,
        rec_embed="SASRec", output_dir=str(artifact_output), save="part",
        lr=args.lr, weight_decay=1e-5, lr_scheduler="cosine",
        lr_decay_min_lr=8e-6, lr_warmup_start_lr=8e-6,
    )
    model = MInterface(**hparams)
    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.enable_input_require_grads()
    prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]
    module = AmazonModule(args, data, model.llama_tokenizer, prompts)
    model.hparams.max_steps = max(
        len(module.trainset) * args.max_epochs //
        (args.accumulate_grad_batches * args.batch_size * max(args.devices, 1)), 1
    )
    callbacks = [EarlyStopping(monitor="metric", mode="max", patience=10, min_delta=0.001)]
    strategy = DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, strategy=strategy, precision="bf16-mixed",
        max_epochs=args.max_epochs, accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks, logger=False, enable_checkpointing=False,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model=model, datamodule=module)
    validation = {str(k): scalar(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("val_") or str(k) == "metric"}
    trainer.test(model=model, datamodule=module)
    test = {str(k): scalar(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("test_") or str(k) == "metric"}
    if trainer.is_global_zero:
        result = {"dataset": args.dataset, "validation": validation, "test": test,
                  "num_users": data.num_users, "num_items": data.num_items,
                  "train_interactions": sum(map(len, data.train_by_user.values())),
                  "train_examples": len(module.trainset), "validation_examples": len(module.valset),
                  "test_examples": len(module.testset), "visible_gpus": torch.cuda.device_count(),
                  "interaction_cache": str(interaction_cache), "rec_model_cache": str(rec_path)}
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print("FINAL_RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
