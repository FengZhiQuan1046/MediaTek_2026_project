#!/usr/bin/env python3
"""Short, measured LLM-SRec training pilot on cached Full Beauty data.

The original LLM-SRec model code is imported unchanged. A short SASRec teacher
is prepared from the same cache because the upstream teacher checkpoint is not
distributed with the repository. This is a pilot, not a reproduction score.
"""
import json
import logging
import os
import pickle
import random
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT / "llm-srec"
VER4 = ROOT / "MediaTek_2026_project" / "ver4"
DATA_FILE = ROOT / "cache" / "mamba_multi_agent_data" / "amazon-all-beauty_90e3e920169a.pkl"
OUT = HERE / "gpu1_runs" / "llm_srec_full_beauty" / "model_output"
sys.path.insert(0, str(VER4))
sys.path.insert(0, str(REPO))


def sync():
    torch.cuda.synchronize()


def main():
    assert torch.cuda.device_count() == 1, "This pilot requires one visible GPU"
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(HERE)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
    start = time.perf_counter()
    with DATA_FILE.open("rb") as file:
        data = pickle.load(file)
    users = sorted(data.train_by_user)
    user_map = {user: index + 1 for index, user in enumerate(users)}
    histories = {user_map[user]: [item + 1 for item in data.train_by_user[user]] for user in users}
    text = {"title": {i + 1: str(s)[:90] for i, s in enumerate(data.item_texts)},
            "description": defaultdict(str), "time": defaultdict(dict)}
    for user, items in histories.items():
        for position, item in enumerate(items):
            text["time"][item][user] = str(1609459200000 + position * 86400000)
    data_dir = HERE / "SeqRec" / "data_FullBeauty"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "text_name_dict.json.gz").open("wb") as file:
        pickle.dump(text, file)
    prepare_seconds = time.perf_counter() - start

    from SeqRec.sasrec.model import SASRec
    teacher_args = Namespace(device="cuda:0", hidden_units=64, maxlen=128,
                             num_blocks=2, num_heads=1, dropout_rate=0.1,
                             nn_parameter=False)
    teacher = SASRec(len(users), data.num_items, teacher_args).cuda()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
    teacher.train()
    teacher_start = time.perf_counter()
    eligible = [u for u in histories if len(histories[u]) >= 3]
    for offset in range(0, min(len(eligible), 384), 128):
        batch_users = eligible[offset:offset + 128]
        seq = np.zeros((len(batch_users), 128), dtype=np.int32)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        for row, user in enumerate(batch_users):
            history = histories[user][-129:]
            seq[row, -len(history) + 1:] = history[:-1]
            pos[row, -len(history) + 1:] = history[1:]
            neg[row, -len(history) + 1:] = np.random.randint(1, data.num_items + 1, len(history) - 1)
        pos_logits, neg_logits = teacher(np.array(batch_users), seq, pos, neg)
        mask = torch.as_tensor(pos != 0, device="cuda:0")
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(pos_logits[mask], torch.ones_like(pos_logits[mask]))
                + torch.nn.functional.binary_cross_entropy_with_logits(neg_logits[mask], torch.zeros_like(neg_logits[mask])))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sync()
    teacher_seconds = time.perf_counter() - teacher_start
    teacher_dir = HERE / "SeqRec" / "sasrec" / "FullBeauty"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    torch.save([teacher.kwargs, teacher.state_dict()], teacher_dir / "pilot_teacher.pth")
    del teacher, optimizer
    torch.cuda.empty_cache()

    # The upstream code hardcodes a Hugging Face model ID without a cache_dir.
    # Bind that ID to the verified local snapshot to avoid a gated HEAD request.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache_roots = [ROOT / "cache", Path("/dataspace/P78123011/cache")]
    snapshots = [p for cache_root in cache_roots
                 for p in (cache_root / "models--meta-llama--Llama-3.2-3B-Instruct" / "snapshots").glob("*")
                 if (p / "model-00001-of-00002.safetensors").is_file()
                 and (p / "model-00002-of-00002.safetensors").is_file()]
    if not snapshots:
        raise FileNotFoundError("Complete Llama-3.2-3B-Instruct snapshot missing from the model caches")
    snapshot = sorted(snapshots)[-1]
    original_model_load = AutoModelForCausalLM.from_pretrained
    original_tokenizer_load = AutoTokenizer.from_pretrained
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    AutoModelForCausalLM.from_pretrained = classmethod(
        lambda cls, path, *a, **kw: original_model_load(str(snapshot) if path == model_id else path, *a, **kw))
    AutoTokenizer.from_pretrained = classmethod(
        lambda cls, path, *a, **kw: original_tokenizer_load(str(snapshot) if path == model_id else path, *a, **kw))
    from models.seqllm_model import llmrec_model
    args = Namespace(device="cuda:0", rec_pre_trained_data="FullBeauty", recsys="sasrec",
                     llm="llama-3b", maxlen=128, nn_parameter=False, token=False,
                     save_dir="overhead", train=True)
    load_start = time.perf_counter()
    model = llmrec_model(args).cuda()
    sync()
    llm_load_seconds = time.perf_counter() - load_start
    model.train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    batch_users = eligible[:2]
    seq = np.zeros((2, 128), dtype=np.int32)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row, user in enumerate(batch_users):
        history = histories[user][-129:]
        seq[row, -len(history) + 1:] = history[:-1]
        pos[row, -len(history) + 1:] = history[1:]
        neg[row, -len(history) + 1:] = np.random.randint(1, data.num_items + 1, len(history) - 1)
    sample = [np.array(batch_users), seq, pos, neg]
    train_start = time.perf_counter()
    model.pre_train_phase2(sample, optimizer, [1, 1, 1, 1])
    sync()
    train_seconds = time.perf_counter() - train_start
    model.eval()
    test_users = [user_map[user] for user in users
                  if user in data.test_target and len(data.train_by_user[user]) >= 3][:129]

    def test_batch(selected):
        test_seq = np.zeros((len(selected), 128), dtype=np.int32)
        test_pos = np.zeros(len(selected), dtype=np.int32)
        inverse = {mapped: original for original, mapped in user_map.items()}
        for row, user in enumerate(selected):
            items = histories[user][-128:]
            test_seq[row, -len(items):] = items
            test_pos[row] = data.test_target[inverse[user]] + 1
        return [np.array(selected), test_seq, test_pos, None, None, None, None]

    # The first call builds LLM-SRec's full-catalog item embedding cache.
    precompute_start = time.perf_counter()
    model.generate_batch(test_batch(test_users[:1]))
    sync()
    precompute_seconds = time.perf_counter() - precompute_start
    # Keep the upstream user encoder and item cache, but score every item to
    # match the full-catalog throughput definition used by the other pilots.
    model.make_candidate = lambda interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None, task="ItemTask": [
        target_item_id, *(item for item in range(1, data.num_items + 1) if item != target_item_id)]
    test_start = time.perf_counter()
    for offset in range(1, len(test_users), 8):
        model.generate_batch(test_batch(test_users[offset:offset + 8]))
    sync()
    test_seconds = time.perf_counter() - test_start
    metrics = dict(model="LLM-SRec", dataset="Full Beauty", physical_gpu_index=1,
                   llm="meta-llama/Llama-3.2-3B-Instruct", llm_dtype="fp16",
                   llm_quantization="8-bit", users=len(users), items=data.num_items,
                   training_batch_size=2, teacher_training_steps=3,
                   data_preparation_seconds=prepare_seconds,
                   teacher_training_seconds=teacher_seconds,
                   llm_load_seconds=llm_load_seconds,
                   llm_training_step_seconds=train_seconds,
                   llm_training_users_per_second=len(batch_users) / train_seconds,
                   full_catalog_precompute_seconds=precompute_seconds,
                   test_seconds=test_seconds,
                   test={"evaluated_users": len(test_users) - 1,
                         "users_per_second": (len(test_users) - 1) / test_seconds,
                         "catalog_size": data.num_items},
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                   caveat="Short training pilot; teacher trained for 3 batches from cached data, not official pretrained teacher; full-catalog item embeddings are cached before timed user evaluation; upstream user encoder with all 4090 items scored.")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
