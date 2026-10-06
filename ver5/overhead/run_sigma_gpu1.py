#!/usr/bin/env python3
"""Short SIGMA run on the project's Full Beauty data with a pure-PyTorch Mamba backend."""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SIGMA_MODEL = WORKSPACE / "SIGMA" / "model"


def install_mamba_backend():
    from mambapy.mamba import Mamba as PureMamba, MambaConfig

    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            self.core = PureMamba(MambaConfig(d_model=d_model, n_layers=1,
                                               d_state=d_state, d_conv=d_conv,
                                               expand_factor=expand, use_cuda=False))

        def forward(self, inputs):
            return self.core(inputs)

    shim = types.ModuleType("mamba_ssm")
    shim.Mamba = Mamba
    sys.modules["mamba_ssm"] = shim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=HERE / "gpu1_runs" / "sigma_full_beauty" / "data")
    parser.add_argument("--output", type=Path, default=HERE / "gpu1_runs" / "sigma_full_beauty" / "model_output")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("exactly one visible GPU is required; launch through profile_gpu1.py")
    torch.cuda.set_device(0)
    install_mamba_backend()
    # RecBole 1.2.0 refers to NumPy aliases removed in NumPy 2.x.
    import numpy as np
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_
    sys.path.insert(0, str(SIGMA_MODEL))
    from gated_mamba import SIGMA
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import init_seed

    args.output.mkdir(parents=True, exist_ok=True)
    overrides = {
        "dataset": "FullBeauty", "data_path": str(args.data_path.resolve()),
        "epochs": args.epochs, "train_batch_size": 2048, "eval_batch_size": 256,
        "eval_step": 1, "stopping_step": 1, "show_progress": False,
        "MAX_ITEM_LIST_LENGTH": 50, "user_inter_num_interval": "[1,inf)",
        "item_inter_num_interval": "[1,inf)", "filter_inter_by_user_or_item": False,
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "group_by": "user", "mode": "full"},
        "checkpoint_dir": str(args.output.resolve()), "gpu_id": "0", "use_gpu": True,
    }
    config = Config(model=SIGMA, dataset="FullBeauty",
                    config_file_list=[str(SIGMA_MODEL / "config.yaml")], config_dict=overrides)
    init_seed(config["seed"], config["reproducibility"])
    started = time.perf_counter()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    dataset_seconds = time.perf_counter() - started
    model = SIGMA(config, train_data.dataset).to(config["device"])
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainer = Trainer(config, model)
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=False, show_progress=False)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_started
    test_users = int(len(test_data.dataset))
    result = dict(model="SIGMA", dataset="FullBeauty", backend="mambapy-1.2.0-pure-pytorch",
                  physical_gpu_index=1, visible_cuda_devices=torch.cuda.device_count(),
                  num_users=dataset.user_num, num_items=dataset.item_num,
                  trainable_parameters=params, dataset_seconds=dataset_seconds,
                  training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                  test={"evaluated_users": test_users,
                        "users_per_second": test_users / evaluation_seconds},
                  best_valid_score=float(best_valid_score),
                  best_valid_result={key: float(value) for key, value in best_valid_result.items()},
                  test_result={key: float(value) for key, value in test_result.items()})
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
