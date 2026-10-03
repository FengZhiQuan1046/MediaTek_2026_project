"""Self-contained LLaRA language-model interface for the Amazon reproduction."""
from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model


class MlpProjector(nn.Module):
    def __init__(self, rec_size: int, llm_size: int):
        super().__init__()
        self.mlp_proj = nn.Sequential(nn.Linear(rec_size, llm_size), nn.GELU(),
                                      nn.Linear(llm_size, llm_size))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp_proj(embeddings)


class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.scheduler = None
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.hparams.llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens({"additional_special_tokens":
            ["[PH]", "[HistoryEmb]", "[CansEmb]", "[ItemEmb]"]})
        self.llama_model = LlamaForCausalLM.from_pretrained(
            self.hparams.llm_path, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_safetensors=True)
        # Only five LLaRA markers are new. Avoid the costly covariance-based
        # initialization introduced by newer Transformers releases.
        self.llama_model.resize_token_embeddings(
            len(self.llama_tokenizer), mean_resizing=False)
        if self.hparams.llm_tuning != "lora":
            raise ValueError("This reproduction supports llm_tuning='lora'")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=self.hparams.lora_r, lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.llama_model = get_peft_model(self.llama_model, config)
        self.load_rec_model(self.hparams.rec_model_path)
        self.projector = MlpProjector(self.hparams.rec_size, self.llama_model.config.hidden_size)

    def load_rec_model(self, rec_model_path):
        raise NotImplementedError

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        return self.projector(self.rec_model.cacu_x(items))

    def wrap_emb(self, batch):
        raise NotImplementedError

    def forward(self, batch):
        tokens = batch["tokens"]
        labels = tokens.input_ids.masked_fill(tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        labels = labels.masked_fill(tokens.token_type_ids[:, 1:] == 0, -100)
        return self.llama_model(inputs_embeds=self.wrap_emb(batch),
                                attention_mask=tokens.attention_mask, labels=labels,
                                return_dict=True, use_cache=False)

    def generate(self, batch):
        ids = self.llama_model.generate(
            inputs_embeds=self.wrap_emb(batch), attention_mask=batch["tokens"].attention_mask,
            temperature=0.8, do_sample=False, num_beams=1, max_new_tokens=64,
            min_new_tokens=1, pad_token_id=self.llama_tokenizer.pad_token_id)
        return [text.strip() for text in self.llama_tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

    def training_step(self, batch, batch_idx):
        for parameter in self.projector.parameters():
            parameter.requires_grad_(not batch["flag"])
        if self.scheduler is not None:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        loss = self(batch).loss
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return [(generated.strip().split("\n")[0], real, candidates)
                for generated, real, candidates in zip(
                    self.generate(batch), batch["correct_answer"], batch["cans_name"])]

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.projector.parameters(), "lr": self.hparams.lr,
             "weight_decay": self.hparams.weight_decay},
            {"params": self.llama_model.parameters(), "lr": self.hparams.lr},
        ])
        self.scheduler = LinearWarmupCosineLRScheduler(
            optimizer, self.hparams.lr, self.hparams.lr_decay_min_lr,
            self.hparams.lr_warmup_start_lr, max(self.trainer.max_steps // 20, 0))
        return optimizer

    @staticmethod
    def calculate_hr1(content):
        valid = correct = 0
        for generated, real, candidates in zip(content["generate"], content["real"], content["cans"]):
            generated = generated.strip().lower()
            matches = [candidate.strip().lower() for candidate in candidates
                       if candidate.strip().lower() in generated]
            if len(matches) == 1:
                valid += 1
                correct += matches[0] == real.strip().lower()
        return valid / len(content["generate"]) if content["generate"] else 0.0, correct / valid if valid else 0.0


class LinearWarmupCosineLRScheduler:
    def __init__(self, optimizer, initial, minimum, warmup_start, warmup_steps):
        self.optimizer = optimizer
        self.initial = initial
        self.minimum = minimum
        self.warmup_start = warmup_start
        self.warmup_steps = warmup_steps

    def step(self, global_step, epoch, max_steps):
        if epoch == 0 and global_step < self.warmup_steps:
            lr = self.warmup_start + (self.initial - self.warmup_start) * global_step / max(self.warmup_steps, 1)
        else:
            lr = self.minimum + (self.initial - self.minimum) * (1 + math.cos(math.pi * global_step / max_steps)) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = lr
