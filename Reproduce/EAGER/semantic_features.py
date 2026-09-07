"""Cached T5 item features used by the semantic stream in upstream EAGER."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import torch
from torch.nn import functional as F


def hashed_features(texts, dimension=768):
    result = torch.zeros(len(texts), dimension)
    for row, text in enumerate(texts):
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())[:256]:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            result[row, value % dimension] += 1.0 if value >> 63 else -1.0
    return F.normalize(result, dim=-1)


@torch.inference_mode()
def load_or_encode(texts, cache_dir, dataset, model_id, batch_size, max_tokens, device, backend="t5"):
    safe = dataset.replace(":", "_").replace("/", "_")
    fingerprint = hashlib.sha1(
        (model_id + "\n" + str(max_tokens) + "\n" + "\n".join(texts)).encode()
    ).hexdigest()[:12]
    artifact = Path(cache_dir) / "eager" / "semantic" / f"{safe}_{fingerprint}.pt"
    if artifact.exists():
        return torch.load(artifact, map_location="cpu", weights_only=True), artifact
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if backend == "hashed":
        features = hashed_features(texts)
    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        chunks = []
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=max_tokens, return_tensors="pt",
            ).to(device)
            encoder = model.get_encoder() if hasattr(model, "get_encoder") else model
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            chunks.append(F.normalize(pooled.float(), dim=-1).cpu())
        features = torch.cat(chunks)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    torch.save(features, artifact)
    return features, artifact
