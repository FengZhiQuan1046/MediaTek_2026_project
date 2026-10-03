"""LLaRA inputs on the exact ver4 split and transition selection."""
from __future__ import annotations

import random
import torch
from torch.utils.data import Dataset

from data_adapter import item_names, make_examples


class AmazonData(Dataset):
    def __init__(self, interaction_data, stage, maxlen=100,
                 max_train_samples=0, eval_user_limit=0, seed=25252):
        self.data = interaction_data
        self.stage = stage
        self.maxlen = maxlen
        self.padding_item_id = interaction_data.num_items
        self.names = item_names(interaction_data)
        maximum = max_train_samples if stage == "train" else eval_user_limit
        self.examples = make_examples(interaction_data, stage, maxlen, maximum, seed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        user, history, target = self.examples[index]
        padded = history + [self.padding_item_id] * (self.maxlen - len(history))
        return {"seq": padded, "seq_name": [self.names[item] for item in history],
                "len_seq": len(history), "item_id": target,
                "correct_answer": self.names[target], "user": user}


class TrainCollater:
    def __init__(self, tokenizer, prompts, train, max_step=1):
        self.tokenizer, self.prompts, self.train = tokenizer, prompts, train
        self.max_step, self.cur_step = max(max_step, 1), 1

    def __call__(self, batch):
        template = random.choice(self.prompts) if self.train else self.prompts[0]
        texts, targets = [], []
        flag = self.train and random.random() >= self.cur_step / self.max_step
        marker = "[PH]" if flag else "[HistoryEmb]"
        for sample in batch:
            text = template.replace("[HistoryHere]", ", ".join(
                name + f" {marker}" for name in sample["seq_name"]
            ))
            texts.append(text); targets.append(sample["correct_answer"])
        self.cur_step += 1
        if self.train:
            tokens = self.tokenizer([[x, y + "\n"] for x, y in zip(texts, targets)],
                                    return_tensors="pt", padding="longest",
                                    add_special_tokens=True, return_attention_mask=True,
                                    return_token_type_ids=True)
            if tokens.token_type_ids.size(1) == tokens.input_ids.size(1):
                leading = torch.zeros((tokens.token_type_ids.size(0), 1),
                                      dtype=tokens.token_type_ids.dtype)
                tokens["token_type_ids"] = torch.cat((leading, tokens.token_type_ids), dim=1)
        else:
            tokens = self.tokenizer(texts, return_tensors="pt", padding="longest",
                                    add_special_tokens=True, return_attention_mask=True)
        result = {"tokens": tokens,
                  "seq": torch.tensor([x["seq"] for x in batch]),
                  "len_seq": torch.tensor([x["len_seq"] for x in batch]),
                  "item_id": torch.tensor([x["item_id"] for x in batch]),
                  "flag": flag, "user": [x["user"] for x in batch]}
        return result
