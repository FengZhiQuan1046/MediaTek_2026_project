"""Unchanged upstream SIGMA layers with a minimal project tensor interface.

RecBole's dataset/trainer is intentionally bypassed. The compatibility base
provides only the padding-ID convention and gather_indexes used by SIGMA.
"""
import ast
from pathlib import Path
import types

import torch
from torch import nn
from mambapy.mamba import Mamba as PureMamba, MambaConfig

MAMBA_BACKEND = 'upstream-SIGMA/mambapy-1.2.0'


class Mamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.core = PureMamba(MambaConfig(d_model=d_model, n_layers=1,
            d_state=d_state, d_conv=d_conv, expand_factor=expand, use_cuda=False))

    def forward(self, inputs):
        return self.core(inputs)


class SequentialRecommender(nn.Module):
    """Only the base attributes referenced by upstream gated_mamba.py."""
    def __init__(self, config, dataset):
        super().__init__()
        self.n_items = dataset.item_num
        self.ITEM_SEQ = 'item_seq'
        self.ITEM_SEQ_LEN = 'item_seq_len'
        self.POS_ITEM_ID = 'pos_item_id'
        self.NEG_ITEM_ID = 'neg_item_id'
        self.ITEM_ID = 'item_id'

    @staticmethod
    def gather_indexes(output, indices):
        return output[torch.arange(output.shape[0], device=output.device), indices]


source = Path(__file__).parent / 'upstream/model/gated_mamba.py'
tree = ast.parse(source.read_text(), filename=str(source))
# Remove only framework imports; every upstream class/method body is intact.
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.ImportFrom) and
    (node.module == 'mamba_ssm' or (node.module or '').startswith('recbole'))
)]
namespace = {'__name__': 'sigma_upstream', 'Mamba': Mamba,
             'SequentialRecommender': SequentialRecommender}
exec(compile(tree, str(source), 'exec'), namespace)
UpstreamSIGMA = namespace['SIGMA']


class SIGMA(UpstreamSIGMA):
    def __init__(self, num_items, hidden_size=64, num_layers=1, dropout=0.2,
                 d_state=32, d_conv=4, expand=2):
        config = dict(hidden_size=hidden_size, num_layers=num_layers,
            dropout_prob=dropout, loss_type='CE', d_state=d_state,
            d_conv=d_conv, expand=expand)
        super().__init__(config, types.SimpleNamespace(item_num=num_items + 1))
        self.num_items = num_items

    def loss(self, sequences, lengths, targets):
        return self.calculate_loss({self.ITEM_SEQ: sequences,
            self.ITEM_SEQ_LEN: lengths, self.POS_ITEM_ID: targets + 1})

    def full_catalog_scores(self, sequences, lengths):
        # Dataset IDs remain zero-based; padding exists only inside model input.
        return self.full_sort_predict({self.ITEM_SEQ: sequences,
            self.ITEM_SEQ_LEN: lengths})[:, 1:]
