"""Standalone Mamba4Rec, adapted from the official RecBole implementation."""
from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
    MAMBA_BACKEND = "mamba_ssm"
except (ImportError, OSError):
    # The CUDA extension is tied closely to the installed torch/CUDA ABI. The
    # pure-PyTorch implementation preserves the Mamba recurrence and works in
    # the shared ver4 environment without replacing its pinned PyTorch build.
    try:
        from mambapy.mamba import Mamba as _PureMamba
        from mambapy.mamba import MambaConfig as _PureMambaConfig

        class Mamba(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand):
                super().__init__()
                config = _PureMambaConfig(
                    d_model=d_model,
                    n_layers=1,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand_factor=expand,
                    use_cuda=False,
                )
                self.core = _PureMamba(config)

            def forward(self, inputs):
                return self.core(inputs)

        MAMBA_BACKEND = "mambapy"
    except (ImportError, OSError):
        Mamba = None
        MAMBA_BACKEND = "unavailable"


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, inputs):
        return self.norm(self.layers(inputs) + inputs)


class MambaLayer(nn.Module):
    def __init__(self, hidden_size, d_state, d_conv, expand, dropout, residual, mixer_factory=None):
        super().__init__()
        mixer = mixer_factory or Mamba
        if mixer is None:
            raise ImportError(
                "No Mamba backend is available; install requirements.txt with "
                "the same Python interpreter used by run.sh"
            )
        self.mamba = mixer(d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = FeedForward(hidden_size, dropout)
        self.residual = residual

    def forward(self, inputs):
        hidden = self.dropout(self.mamba(inputs))
        hidden = self.norm(hidden + inputs if self.residual else hidden)
        return self.ffn(hidden)


class Mamba4Rec(nn.Module):
    def __init__(self, num_items, hidden_size=64, num_layers=1, dropout=0.2,
                 d_state=32, d_conv=4, expand=2, mixer_factory=None):
        super().__init__()
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MambaLayer(hidden_size, d_state, d_conv, expand, dropout,
                       residual=num_layers > 1, mixer_factory=mixer_factory)
            for _ in range(num_layers)
        ])
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def encode(self, sequences, lengths):
        hidden = self.input_norm(self.dropout(self.item_embedding(sequences)))
        for layer in self.layers:
            hidden = layer(hidden)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[rows, lengths - 1]

    def full_catalog_scores(self, sequences, lengths):
        # Padding is excluded; returned columns are zero-based ver4 item IDs.
        return self.encode(sequences, lengths) @ self.item_embedding.weight[1:].T

    def loss(self, sequences, lengths, targets):
        return nn.functional.cross_entropy(self.full_catalog_scores(sequences, lengths), targets)


class LargeMamba4Rec(nn.Module):
    """Mamba4Rec item objective backed by the pretrained Mamba2 2.7B stack."""

    def __init__(self, num_items, model_id, cache_dir, dtype="bfloat16", device_map=None):
        super().__init__()
        from transformers import AutoModel

        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.backbone = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtypes[dtype],
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        self.is_model_parallel = device_map is not None
        self.backbone.gradient_checkpointing_enable()
        hidden_size = self.backbone.config.hidden_size
        parameter = next(self.backbone.parameters())
        self.item_embedding = nn.Embedding(
            num_items + 1,
            hidden_size,
            padding_idx=0,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, sequences, lengths):
        input_device = self.item_embedding.weight.device
        model_sequences = sequences.to(input_device)
        attention_mask = model_sequences.ne(0)
        outputs = self.backbone(
            inputs_embeds=self.item_embedding(model_sequences),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        output_device = outputs.last_hidden_state.device
        rows = torch.arange(sequences.size(0), device=output_device)
        return outputs.last_hidden_state[rows, lengths.to(output_device) - 1]

    def full_catalog_scores(self, sequences, lengths):
        states = self.encode(sequences, lengths)
        catalog = self.item_embedding.weight[1:].to(states.device)
        return (states @ catalog.T).to(sequences.device)

    def loss(self, sequences, lengths, targets):
        return nn.functional.cross_entropy(
            self.full_catalog_scores(sequences, lengths).float(), targets
        )
