#!/usr/bin/env python3
"""Export the project's Full Beauty chronology to RecBole's atomic format."""
import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
VER4 = WORKSPACE / "MediaTek_2026_project" / "ver4"
sys.path.insert(0, str(VER4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "gpu1_runs" / "sigma_full_beauty" / "data")
    args = parser.parse_args()
    source = WORKSPACE / "cache" / "mamba_multi_agent_data" / "amazon-all-beauty_90e3e920169a.pkl"
    with source.open("rb") as file:
        data = pickle.load(file)
    target_dir = args.output / "FullBeauty"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "FullBeauty.inter"
    with target.open("w") as file:
        file.write("user_id:token\titem_id:token\ttimestamp:float\n")
        interactions = 0
        for user, train in sorted(data.train_by_user.items()):
            full_sequence = list(train)
            if user in data.valid_target:
                full_sequence.append(data.valid_target[user])
            if user in data.test_target:
                full_sequence.append(data.test_target[user])
            for position, item in enumerate(full_sequence, 1):
                file.write(f"{user}\t{item}\t{position}\n")
                interactions += 1
    manifest = dict(source=str(source), output=str(target), users=data.num_users,
                    items=data.num_items, interactions=interactions,
                    ordering="training interactions followed by validation and test targets; timestamp is within-user ordinal")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
