"""Pretrain the bundled LLaRA-style SASRec class for an Amazon catalog."""
from __future__ import annotations

from pathlib import Path
import random

import torch
from tqdm.auto import tqdm

from sasrec import SASRec


def train_rec_model(data, output, device, epochs=10, batch_size=256,
                    hidden_size=64, maxlen=10, learning_rate=1e-3, seed=1234):
    output = Path(output)
    if output.exists():
        return output
    rng = random.Random(seed)
    eligible = [seq for seq in data.train_by_user.values() if len(seq) > 1]
    if not eligible:
        raise ValueError("SASRec pretraining requires a user with at least two training items")
    model = SASRec(hidden_size, data.num_items, maxlen, 0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = max(1, sum(len(seq) - 1 for seq in eligible) // batch_size)
    for epoch in range(epochs):
        model.train()
        for _ in tqdm(range(steps), desc=f"LLaRA SASRec {epoch + 1}/{epochs}"):
            states = torch.full((batch_size, maxlen), data.num_items,
                                dtype=torch.long, device=device)
            lengths = torch.empty(batch_size, dtype=torch.long, device=device)
            targets = torch.empty(batch_size, dtype=torch.long, device=device)
            for row in range(batch_size):
                sequence = rng.choice(eligible); end = rng.randrange(1, len(sequence))
                history = sequence[max(0, end - maxlen):end]
                states[row, :len(history)] = torch.tensor(history, device=device)
                lengths[row] = len(history); targets[row] = sequence[end]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(states, lengths), targets)
            loss.backward(); optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "hidden_size": hidden_size, "item_num": data.num_items,
                "state_size": maxlen, "state_dict": model.cpu().state_dict()}, output)
    return output
