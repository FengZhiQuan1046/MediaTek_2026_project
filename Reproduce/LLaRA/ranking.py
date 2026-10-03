"""Full-catalog LLaRA scoring with ver4's rank and metric protocol."""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def full_catalog_scores(model, token_ids, token_masks, prompt_embeddings, chunk_size=8,
                        progress=None):
    """Score all item names against one history prompt, reusing its KV cache.

    Sort by continuation length to avoid padding every item to the longest
    catalog name. Crop the repeated prompt cache after each chunk instead of
    copying the complete cache for each of thousands of chunks.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    device = prompt_embeddings.device
    count = token_ids.size(0)
    if not count or int(token_masks.sum(1).min()) < 2:
        raise ValueError("Catalog must contain names with at least two tokens")
    lengths = token_masks.sum(1)
    order = torch.argsort(lengths, stable=True)
    prefix_length = prompt_embeddings.size(0)
    prefix = model(inputs_embeds=prompt_embeddings.unsqueeze(0), use_cache=True,
                   logits_to_keep=1)
    first_logits = prefix.logits[:, -1].float()
    cache = prefix.past_key_values
    if not hasattr(cache, "crop") or not hasattr(cache, "batch_repeat_interleave"):
        raise TypeError("Full-catalog ranking requires a reusable Transformers DynamicCache")
    cache.batch_repeat_interleave(chunk_size)
    scores = torch.empty(count, device=device, dtype=torch.float32)
    for start in range(0, count, chunk_size):
        indices = order[start:start + chunk_size]
        actual_size = indices.numel()
        width = int(lengths[indices].max())
        ids = token_ids[indices, :width].to(device)
        mask = token_masks[indices, :width].to(device)
        if actual_size < chunk_size:
            # Keep the KV cache batch size fixed for the final partial chunk.
            extra = chunk_size - actual_size
            ids = torch.cat((ids, ids[-1:].expand(extra, -1)), dim=0)
            mask = torch.cat((mask, mask[-1:].expand(extra, -1)), dim=0)
        attention = torch.cat((torch.ones((chunk_size, prefix_length),
                                          device=device, dtype=mask.dtype), mask[:, :-1]), dim=1)
        continuation = model(input_ids=ids[:, :-1], attention_mask=attention,
                             past_key_values=cache, use_cache=False)
        logits = torch.cat((first_logits.expand(chunk_size, -1).unsqueeze(1),
                            continuation.logits.float()), dim=1)
        token_scores = logits.gather(2, ids.unsqueeze(-1)).squeeze(-1)
        token_scores -= torch.logsumexp(logits, dim=-1)
        chunk_scores = (token_scores * mask).sum(1) / mask.sum(1)
        scores[indices.to(device)] = chunk_scores[:actual_size]
        cache.crop(prefix_length)
        if progress is not None and (start // chunk_size) % 32 == 0:
            progress(min(start + chunk_size, count), count)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite full-catalog LLaRA scores")
    return scores


def rank_like_ver4(scores, history, target, priors=None, popularity_alpha=0.0,
                    transition_beta=0.0):
    """Apply ver4 priors and seen-item mask, then count ties as ver4 does."""
    scores = scores.float().clone()
    if priors is not None:
        scores += popularity_alpha * priors.popularity.to(scores.device)
        if transition_beta != 0.0 and history:
            transitions = priors.transitions.get(history[-1], {})
            if transitions:
                ids = list(transitions)
                values = torch.tensor(list(transitions.values()), device=scores.device,
                                      dtype=scores.dtype)
                scores[ids] += transition_beta * values
    seen = set(history) - {int(target)}
    if seen:
        scores[list(seen)] = -torch.inf
    return int((scores >= scores[int(target)]).sum().item())


def ranking_metrics(ranks):
    ranks = list(ranks)
    if not ranks or min(ranks) < 1:
        raise ValueError("Expected positive one-based ranks")
    result = {}
    for cutoff in (5, 10):
        hits = sum(rank <= cutoff for rank in ranks) / len(ranks)
        result[f"recall@{cutoff}"] = hits
        result[f"hit@{cutoff}"] = hits
        result[f"ndcg@{cutoff}"] = sum(1 / math.log2(rank + 1)
                                       for rank in ranks if rank <= cutoff) / len(ranks)
    return result
