"""Pretrain the upstream LLaRA SASRec class for an Amazon catalog."""
from __future__ import annotations

from pathlib import Path
import random
import sys

import torch
from tqdm.auto import tqdm

UPSTREAM = Path(__file__).resolve().parents[3] / "LLaRA"
for path in (UPSTREAM,):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from recommender.A_SASRec_final_bce_llm import SASRec  # noqa: E402


def train_rec_model(data, output, device, epochs=10, batch_size=256,
                    hidden_size=64, maxlen=10, learning_rate=1e-3, seed=1234):
    output = Path(output)
    if output.exists():
        return output
    rng = random.Random(seed)
    eligible = [seq for seq in data.train_by_user.values() if len(seq) > 1]
    model = SASRec(hidden_size, data.num_items, maxlen, 0.1, device).to(device)
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
    torch.save(model.cpu(), output)
    return output
