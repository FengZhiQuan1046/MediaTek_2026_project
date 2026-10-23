"""Use the common baseline tensor helpers; no alternate preprocessing or split."""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    'sigma_common_adapter', Path(__file__).resolve().parents[1] / 'Mamba4Rec/data_adapter.py'
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
evaluation_batch = common.evaluation_batch
load_data_cached = common.load_data_cached
mask_seen_items = common.mask_seen_items
ranking_metrics = common.ranking_metrics
sample_prefix_batch = common.sample_prefix_batch
