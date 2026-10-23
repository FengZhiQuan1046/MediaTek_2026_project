"""SPC vs SLC on two target-conditioned older-history similarity cohorts.

SW4/P32, all observed events; no checkpoints. Cohorts are diagnostic
(post hoc target-conditioned), never used to train or select checkpoints.
Item cosine uses frozen text vectors. Preference cosine uses soft assignments
to 32 independent prototypes fitted to training-interaction items only.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm

from src import rq5
from src import train_mamba_rl as train
from src.rq5_sim_model import SLCRecommender

COHORTS = ('item_similarity', 'preference_similarity', 'intersection', 'item_only', 'preference_only')


@contextlib.contextmanager
def slc_training():
    """Use long-item specialist in the second trainer slot, only in this process."""
    original = (train.MultiAgentMambaRecommender, train.preference_auxiliary_losses,
                train.preference_contrastive_loss)
    train.MultiAgentMambaRecommender = SLCRecommender
    train.preference_auxiliary_losses = lambda *a, **k: (0.,) * 5
    train.preference_contrastive_loss = lambda *a, **k: 0.
    try:
        yield
    finally:
        (train.MultiAgentMambaRecommender, train.preference_auxiliary_losses,
         train.preference_contrastive_loss) = original


def train_variant(args, variant):
    directory = args.output_dir / variant
    options = rq5.training_options(args, args.subset, 'full', directory)
    options += ['--experiment-note', f'RQ5 similarity {variant}; SW4 P32; full-history unlimited; validation-selected']
    if variant == 'SLC':
        # Explicit zeros also make the saved training configuration auditable.
        for coefficient in ('preference-coef', 'preference-transition-coef',
                            'preference-balance-coef', 'preference-separation-coef',
                            'preference-sharpness-coef', 'preference-contrastive-coef'):
            options += ['--' + coefficient, '0']
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *options]
        context = slc_training() if variant == 'SLC' else contextlib.nullcontext()
        with context, rq5.read_only_source_cache(args.output_dir / 'cache'), rq5.capture_final_model() as captured:
            train.main()
        if not captured:
            raise RuntimeError('Missing validation-selected final model')
        # Correct new-run metadata for the long-slot compatibility interface.
        score_path = directory / 'scores.json'
        report = json.loads(score_path.read_text())
        report['rq5_sim_architecture'] = variant
        if variant == 'SLC':
            report['model']['recommender'] = 'SLCRecommender'
            report['model']['agents'] = ['short', 'long_item']
            report['model']['ablation']['USE_PREFERENCE'] = 0
            report['model']['ablation']['USE_LONG'] = 1
            report['model']['long_history_limit'] = report['model'].pop('preference_history_limit')
            report['compatibility_note'] = "Trainer config/output slot named 'preference' carries LONG; prototype, transition and preference-contrastive losses disabled. Coordinator context held constant."
        score_path.write_text(json.dumps(report, indent=2))
        return captured
    finally:
        sys.argv = old_argv


def fit_proxy(features, data, count, temperature, fit_limit, seed):
    """No held-out events, ranks, or trained recommender representations enter fit."""
    train_ids = np.unique(np.fromiter((i for h in data.train_by_user.values() for i in h), dtype=np.int64))
    valid_vectors = features.norm(dim=-1).numpy() > 0
    train_ids = train_ids[valid_vectors[train_ids]]
    rng = np.random.default_rng(seed)
    if len(train_ids) > fit_limit:
        train_ids = np.sort(rng.choice(train_ids, fit_limit, replace=False))
    if len(train_ids) < count:
        raise ValueError(f'Need at least {count} nonzero training items for preference proxy')
    normalized = F.normalize(features, dim=-1)
    fit_features = normalized[torch.from_numpy(train_ids)].numpy()
    if len(np.unique(fit_features, axis=0)) < count:
        raise ValueError('Fewer distinct training text vectors than proxy prototypes')
    clusterer = MiniBatchKMeans(n_clusters=count, random_state=seed,
        batch_size=min(1024, len(train_ids)), n_init=3, max_iter=100)
    clusterer.fit(fit_features)
    centers = F.normalize(torch.from_numpy(clusterer.cluster_centers_), dim=-1)
    assignments = torch.empty((len(features), count))
    for i in range(0, len(features), 4096):
        assignments[i:i+4096] = torch.softmax(normalized[i:i+4096] @ centers.T / temperature, dim=-1)
    # Cosine of soft preference distributions, normalized independently of text.
    return normalized, F.normalize(assignments, dim=-1), valid_vectors, centers.numpy(), train_ids


def history_similarities(history, target, item_vectors, preference_vectors, valid_vectors, rng, window=4):
    """Older > recent maxima; ignores zero-text vectors and preserves chronology."""
    history = np.asarray(history, dtype=np.int64)
    if len(history) <= window or not valid_vectors[target]:
        return None
    split = len(history) - window
    older = np.flatnonzero(valid_vectors[history[:split]])
    recent = split + np.flatnonzero(valid_vectors[history[split:]])
    if not len(older) or not len(recent):
        return None
    ids = torch.from_numpy(history)
    item = (item_vectors[ids] @ item_vectors[target]).numpy()
    preference = (preference_vectors[ids] @ preference_vectors[target]).numpy()
    result = {'history_length': len(history), 'older_count': split,
              'target_in_older': bool(target in history[:split]),
              'target_in_recent': bool(target in history[split:])}
    # Equal-count sensitivity removes the older segment's opportunity-count advantage.
    n = min(len(older), len(recent))
    equal_older = np.sort(rng.choice(older, n, replace=False))
    equal_recent = np.sort(rng.choice(recent, n, replace=False))
    for name, similarity in [('item', item), ('preference', preference)]:
        for segment, positions in [('older', older), ('recent', recent)]:
            position = int(positions[np.argmax(similarity[positions])])
            result[f'{name}_{segment}_max'] = float(similarity[position])
            result[f'{name}_{segment}_item'] = int(history[position])
            result[f'{name}_{segment}_lag'] = len(history) - position
        result[f'{name}_gap'] = result[f'{name}_older_max'] - result[f'{name}_recent_max']
        result[f'{name}_equal_count_gap'] = float(similarity[equal_older].max() - similarity[equal_recent].max())
    return result


def test_history(data, user):
    return list(data.train_by_user[user]) + [data.valid_target[user]]


def select_cohorts(features, data, args):
    items, preferences, valid, centers, train_ids = fit_proxy(features, data, 32,
        args.proxy_temperature, args.proxy_fit_limit, args.seed)
    np.savez_compressed(args.output_dir / 'preference_proxy.npz', centers=centers, fit_item_ids=train_ids)
    users = sorted(data.test_target)
    if args.analysis_user_limit:
        users = users[:args.analysis_user_limit]
    records, selected = [], []
    rng = np.random.default_rng(args.seed + 702)
    for user in tqdm(users, desc='RQ5 similarity cohort screening', unit='user'):
        record = history_similarities(test_history(data, user), data.test_target[user],
                                     items, preferences, valid, rng)
        if record is None:
            continue
        record.update(user_id=user, target_item_id=data.test_target[user],
            filtered_full_sequence_length=len(data.train_by_user[user])+2)
        records.append(record)
        selected.append(user)
    if not records:
        raise ValueError('No eligible test users with valid older and recent text vectors')
    arrays = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    arrays['users'] = np.asarray(selected, dtype=np.int64)
    arrays['target'] = arrays['target_item_id']
    arrays['item_selected'] = arrays['item_gap'] > args.minimum_gap
    arrays['preference_selected'] = arrays['preference_gap'] > args.minimum_gap
    return arrays, {'total_test_users': len(data.test_target), 'screened_users': len(users),
        'eligible_users': len(selected), 'proxy_fit_items': len(train_ids),
        'proxy_fit_item_ids_sha256': hashlib.sha256(train_ids.tobytes()).hexdigest()}


@torch.inference_mode()
def evaluate_users(captured, users, args):
    model, data = captured['model'], captured['data']
    model.eval()
    with train.amp_context(args.device):
        graph = model.graph_item_vectors()
        items = model.project_all(graph)
    ranks = np.empty(len(users), dtype=np.int64)
    weights = np.empty((len(users), 2))
    for i in tqdm(range(0, len(users), args.eval_batch_size), desc='RQ5 cohort evaluation', unit='batch'):
        batch = [int(u) for u in users[i:i+args.eval_batch_size]]
        histories, lengths, raw = train.evaluation_history_batch(data, batch, 'test', 0, args.device)
        with train.amp_context(args.device):
            output = model.full_catalog_scores(histories, lengths, items, graph)
        sl = slice(i, i+len(batch))
        ranks[sl] = rq5.masked_ranks(output['coordinator'], data, batch, raw, captured)
        weights[sl] = output['weights'].float().cpu().numpy()
    return ranks, weights


def masks(arrays):
    item, pref = arrays['item_selected'], arrays['preference_selected']
    return {'item_similarity': item, 'preference_similarity': pref,
            'intersection': item & pref, 'item_only': item & ~pref, 'preference_only': pref & ~item}


def comparison(mask, spc, slc, bootstrap, seed):
    n = int(mask.sum())
    out = {'users': n, 'SPC': None, 'SLC': None, 'SPC_minus_SLC': None}
    if n:
        out['SPC'] = dict(zip(rq5.METRICS, spc[mask].mean(0).tolist()))
        out['SLC'] = dict(zip(rq5.METRICS, slc[mask].mean(0).tolist()))
        out['SPC_minus_SLC'] = {}
        for k, metric in enumerate(rq5.METRICS):
            diff = (spc[:, k] - slc[:, k])[mask]
            samples = rq5.bootstrap_means(diff, bootstrap, np.random.default_rng(seed+k))
            out['SPC_minus_SLC'][metric] = {'mean': float(diff.mean()), 'ci95': rq5.interval(samples)}
    return out


def write_results(arrays, screening, parameter_counts, args):
    spc, slc = (rq5.rank_metrics(arrays[f'{v}_rank']) for v in ('SPC', 'SLC'))
    population = len(arrays['users'])
    summary = {'configuration': vars(args).copy(), 'screening': screening,
        'parameter_counts': parameter_counts,
        'selection': {'window': 4, 'history_cap': 0, 'rule': 'older maximum cosine > recent maximum cosine + minimum_gap',
            'minimum_gap': args.minimum_gap, 'item_representation': 'Frozen cached Mamba text vectors; no graph or learned projection',
            'preference_representation': 'Cosine of soft assignments to independent training-item MiniBatchKMeans prototypes',
            'proxy_count': 32, 'proxy_temperature': args.proxy_temperature, 'proxy_seed': args.seed,
            'eligible': 'Nonzero target text vector and at least one usable vector in both older and recent history',
            'overlap': 'Item and preference cohorts may overlap; intersection/exclusive cohorts reported separately',
            'equal_count_sensitivity': 'Sample min(usable older, usable recent) positions from each segment without replacement; same positions for both similarities',
            'diagnostic_only': 'Target-conditioned test screening; does not establish causal use of older information or a deployable cohort.'},
        'architectures': {'SPC': 'Short + preference-transition + coordinator + shared GCN',
            'SLC': 'Short + full-history item selective-state agent (initial timescale .08) + same coordinator + shared GCN; no prototypes or preference auxiliaries, constant coordinator context'},
        'id_space': 'Filtered-dataset integer user/item indices; lag1 is last observed event; length strata use capped observed test history',
        'smoke_test_random_features': args.skip_mamba,
        'cohorts': {}, 'equal_count_sensitivity': {}}
    for ci, (name, mask) in enumerate(masks(arrays).items()):
        entry = comparison(mask, spc, slc, args.bootstrap, args.seed + 100*ci)
        entry['fraction_of_all_test_users'] = int(mask.sum()) / screening['total_test_users']
        entry['fraction_of_eligible_users'] = int(mask.sum()) / population
        repeated = arrays['target_in_older'] | arrays['target_in_recent']
        entry['nonrepeat_targets'] = comparison(mask & ~repeated, spc, slc, args.bootstrap, args.seed+1000+ci)
        entry['repeat_targets'] = comparison(mask & repeated, spc, slc, args.bootstrap, args.seed+2000+ci)
        entry['history_length_strata'] = {}
        for lo, hi in [(5,10),(11,20),(21,50),(51,100)]:
            length_mask = (arrays['history_length']>=lo) & (arrays['history_length']<=hi)
            entry['history_length_strata'][f'{lo}-{hi}'] = comparison(mask & length_mask, spc, slc, args.bootstrap, args.seed+3000+ci+lo)
        summary['cohorts'][name] = entry
        # Separate streaming JSON for each requested cohort, with per-user paired outcomes.
        destination = args.output_dir / f'{name}.json'
        with destination.open('w') as stream:
            stream.write(json.dumps({'cohort': name, 'summary': entry}, allow_nan=False)[:-1]+', "records": [\n')
            first = True
            for i in np.flatnonzero(mask):
                row = {key: value[i].item() for key,value in arrays.items()
                       if value.ndim==1 and key not in ('SPC_rank','SLC_rank')}
                row['SPC'] = {'rank': int(arrays['SPC_rank'][i]), **dict(zip(rq5.METRICS, spc[i].tolist())),
                              'short_weight': float(arrays['SPC_weights'][i,0]), 'preference_weight': float(arrays['SPC_weights'][i,1])}
                row['SLC'] = {'rank': int(arrays['SLC_rank'][i]), **dict(zip(rq5.METRICS, slc[i].tolist())),
                              'short_weight': float(arrays['SLC_weights'][i,0]), 'long_weight': float(arrays['SLC_weights'][i,1])}
                if not first: stream.write(',\n')
                stream.write(json.dumps(row, allow_nan=False))
                first = False
            stream.write('\n]}\n')
    for k in ('item','preference'):
        selected = arrays[f'{k}_equal_count_gap'] > args.minimum_gap
        summary['equal_count_sensitivity'][k] = comparison(selected, spc, slc, args.bootstrap, args.seed+5000)
    summary['eligible_population'] = comparison(np.ones(population,dtype=bool), spc, slc, args.bootstrap, args.seed+6000)
    # Full test scores from each restored model are also available in SPC/SLC scores.json.
    (args.output_dir/'summary.json').write_text(json.dumps(summary, default=str, indent=2, allow_nan=False))
    np.savez_compressed(args.output_dir/'cohort_arrays.npz', **arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument('--subset', choices=rq5.SUBSETS, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, default=root.parents[1]/'cache')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--short-window', type=int, choices=[4], default=4)
    p.add_argument('--preference-count', type=int, choices=[32], default=32)
    p.add_argument('--mamba-model-id', default='state-spaces/mamba-130m-hf')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--encode-batch-size', type=int, default=32)
    p.add_argument('--specialist-lr', type=float, default=1e-4)
    p.add_argument('--joint-epochs', type=int, choices=[0], default=0)
    p.add_argument('--joint-lr', type=float, default=5e-5)
    p.add_argument('--rl-coef', type=float, default=.1)
    p.add_argument('--rl-topk', type=int, default=10)
    p.add_argument('--seed', type=int, default=25252)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--proxy-temperature', type=float, default=.2)
    p.add_argument('--proxy-fit-limit', type=int, default=20000)
    p.add_argument('--minimum-gap', type=float, default=0., help='Strict gap, fixed before evaluation; default older > recent.')
    p.add_argument('--analysis-user-limit', type=int, default=0, help='Debug only; 0 screens all test users.')
    p.add_argument('--max-transitions', type=int, default=0, help='Debug only; 0 uses subset training defaults.')
    p.add_argument('--skip-mamba', action='store_true', help='Smoke test only; random frozen features.')
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.eval_batch_size,args.encode_batch_size,args.bootstrap,args.rl_topk)<1:
        p.error('Counts must be positive')
    if args.proxy_fit_limit<32 or args.proxy_temperature<=0 or args.minimum_gap<0 or min(args.analysis_user_limit,args.max_transitions)<0:
        p.error('Invalid proxy settings, margin or debug limits')
    if min(args.specialist_lr,args.joint_lr)<=0 or args.rl_coef<0:
        p.error('Learning rates must be positive and RL coefficient nonnegative')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    captured = train_variant(args, 'SPC')
    features = captured['model'].item_features.detach().float().cpu()
    arrays, screening = select_cohorts(features, captured['data'], args)
    parameter_counts = {'SPC': captured['model'].agent_parameter_counts()}
    arrays['SPC_rank'], arrays['SPC_weights'] = evaluate_users(captured, arrays['users'], args)
    np.savez_compressed(args.output_dir/'SPC_analysis.npz', **arrays)
    captured.clear()
    del features
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    captured = train_variant(args, 'SLC')
    for i,user in enumerate(arrays['users']):
        assert captured['data'].test_target[int(user)] == int(arrays['target'][i])
        assert len(test_history(captured['data'], int(user))) == arrays['history_length'][i]
    parameter_counts['SLC'] = captured['model'].agent_parameter_counts()
    arrays['SLC_rank'], arrays['SLC_weights'] = evaluate_users(captured, arrays['users'], args)
    write_results(arrays, screening, parameter_counts, args)
    print(f'RQ5 similarity complete: {args.output_dir / "summary.json"}')


if __name__=='__main__': main()
