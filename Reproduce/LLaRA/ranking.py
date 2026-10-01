"""Candidate ranking for the generative LLaRA model."""

import math

import torch


@torch.no_grad()
def candidate_scores(model, tokenizer, prompt_embeddings, candidate_names):
    """Mean token log probability of each candidate name followed by a newline.

    The model sees the same prompt and candidate list as it does for generation.
    The prompt KV cache is shared by all candidate continuations.
    """
    if not candidate_names:
        raise ValueError("Candidate list is empty")
    device = prompt_embeddings.device
    encoded = tokenizer(
        [name + "\n" for name in candidate_names],
        add_special_tokens=False, padding=True, return_tensors="pt",
    ).to(device)
    ids = encoded.input_ids
    lengths = encoded.attention_mask.sum(dim=1)
    if int(lengths.min()) < 2:
        raise ValueError("Candidate name must contain at least two tokens")

    prefix = model(
        inputs_embeds=prompt_embeddings.unsqueeze(0), use_cache=True,
        logits_to_keep=1,
    )
    cache = prefix.past_key_values
    cache.batch_repeat_interleave(len(candidate_names))
    continuation_mask = encoded.attention_mask[:, :-1]
    attention_mask = torch.cat(
        (
            torch.ones((len(candidate_names), prompt_embeddings.size(0)),
                       device=device, dtype=continuation_mask.dtype),
            continuation_mask,
        ),
        dim=1,
    )
    continuation = model(
        input_ids=ids[:, :-1], attention_mask=attention_mask,
        past_key_values=cache, use_cache=False,
    )
    logits = torch.cat(
        (prefix.logits[:, -1].expand(len(candidate_names), -1).unsqueeze(1),
         continuation.logits),
        dim=1,
    ).float()
    token_log_probs = logits.gather(2, ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs -= torch.logsumexp(logits, dim=-1)
    mask = torch.arange(ids.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
    scores = token_log_probs.masked_fill(~mask, 0).sum(dim=1) / lengths
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite candidate ranking score")
    return scores


def ranking_metrics(ranks):
    """One held-out relevant item per user; ranks are one-based."""
    ranks = list(ranks)
    if not ranks or min(ranks) < 1:
        raise ValueError("Expected positive one-based ranks")
    return {
        f"{name}@{cutoff}": sum(
            (1 / math.log2(rank + 1) if name == "ndcg" else 1)
            if rank <= cutoff else 0
            for rank in ranks
        ) / len(ranks)
        for name in ("ndcg", "recall")
        for cutoff in (5, 10)
    }
