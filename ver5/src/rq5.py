"""Isolated RQ5 training/analysis entry point; existing source files are read only.

Groups use the FULL model's short-head entropy, with validation tertile thresholds.
Entropy excludes observed items without consulting the held-out target; ranking
uses the existing evaluator's priors, seen-item masking and pessimistic ties.
The rho intervention changes only coordinator context, not preference scoring.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import gc
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from src import train_mamba_rl as train

GROUPS = ('low', 'medium', 'high')
METRICS = ('ndcg@5', 'ndcg@10', 'recall@5', 'recall@10')
SUBSETS = {
    'Full_Beauty': ('amazon-all-beauty', 250, 500000, 64, -.25, 4.),
    'Baby_Products': ('amazon:Baby_Products', 6000, 1500000, 192, .30, .5),
    'Sports_and_Outdoors': ('amazon-sports-and-outdoors', 8000, 1000000, 256, .35, .5),
    'Toys_and_Games': ('amazon-toys-and-games', 6000, 1500000, 192, .30, .5),
}


def rank_metrics(ranks):
    ranks = np.asarray(ranks)
    return np.stack([np.where(ranks <= 5, 1 / np.log2(ranks + 1), 0),
                     np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0),
                     (ranks <= 5).astype(float), (ranks <= 10).astype(float)], axis=-1)


def assign_groups(entropy, thresholds):
    # Ties stay together: <= lower is low, > upper is high. Empty groups are explicit.
    lo, hi = thresholds
    entropy = np.asarray(entropy)
    return np.where(entropy <= lo, 0, np.where(entropy <= hi, 1, 2))


def bootstrap_means(values, repeats, rng):
    """Exact ordinary user-bootstrap for the marginal mean, via outcome counts.

    For paired comparisons values must be per-user metric differences. Counting
    repeated outcomes avoids allocating repeats x users samples on large subsets.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    outcomes, counts = np.unique(values, return_counts=True)
    samples = np.empty(repeats)
    for i in range(repeats):
        samples[i] = rng.multinomial(len(values), counts / counts.sum()) @ outcomes / len(values)
    return samples


def interval(samples):
    return None if samples is None else np.quantile(samples, [.025, .975]).tolist()


def entropy_from_scores(scores, histories):
    # No target IDs enter this function: uncertainty is prediction-time information.
    logits = scores.float().clone()
    for row, history in enumerate(histories):
        if history:
            logits[row, list(set(history))] = -torch.inf
    eligible = torch.isfinite(logits).sum(-1)
    if (eligible == 0).any():
        raise ValueError('No unobserved catalog items available for uncertainty')
    logp = torch.log_softmax(logits, dim=-1)
    probability = logp.exp()
    entropy = -(probability * torch.where(torch.isfinite(logp), logp, 0)).sum(-1)
    return torch.where(eligible > 1, entropy / eligible.clamp_min(2).float().log(), 0).clamp(0, 1)


def coordinator_with_rho(model, states, rho):
    # Match encode_states exactly, including mixed-precision arithmetic.
    predicted = states[5].clamp_min(1e-8)
    entropy = -(predicted * predicted.log()).sum(-1) / math.log(predicted.shape[-1])
    rho = rho.to(dtype=states[7].dtype)
    context = torch.stack((rho, 1.0 - entropy), dim=-1)
    coordinator, weights = model.coordinator(states[0], states[2], context)
    modified = list(states)
    modified[1], modified[3], modified[7] = coordinator, weights, rho
    return tuple(modified)


def masked_ranks(scores, data, users, histories, options):
    scores = scores.float().clone()
    priors = options.get('priors')
    if priors is not None:
        scores += options.get('popularity_alpha', 0.) * priors.popularity.unsqueeze(0)
    gold = torch.tensor([data.test_target[u] for u in users], device=scores.device)
    beta = options.get('transition_beta', 0.)
    for row, history in enumerate(histories):
        if priors is not None and beta and history:
            transitions = priors.transitions.get(history[-1], {})
            if transitions:
                ids = list(transitions)
                scores[row, ids] += beta * torch.tensor(list(transitions.values()), device=scores.device)
        seen = set(history) - {int(gold[row])}
        if seen:
            scores[row, list(seen)] = -torch.inf
    return (scores >= scores.gather(1, gold[:, None])).sum(-1).cpu().numpy()


@contextlib.contextmanager
def capture_final_model():
    """Process-local hook, restored even on failure; no source files are patched."""
    original = train.evaluate
    signature = inspect.signature(original)
    captured = {}

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments['split'] == 'test':
            captured.clear()
            captured.update(bound.arguments)
        return result

    train.evaluate = evaluate
    try:
        yield captured
    finally:
        train.evaluate = original


@contextlib.contextmanager
def read_only_source_cache(private_cache):
    """Read existing common caches; publish cache misses only in this new RQ5 run."""
    original_data = train.load_recommendation_data_cached
    original_text = train.load_or_encode_text

    def load_data(options, logger):
        identity = {
            'dataset': options.dataset,
            'data_path': str(Path(options.data_path).resolve()) if options.data_path else None,
            'max_events': options.max_events, 'min_rating': options.min_rating,
            'min_interactions': train.MIN_INTERACTIONS, 'schema_version': 2,
        }
        digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        safe = options.dataset.replace(':', '_').replace('/', '_')
        source = Path(options.cache_dir) / 'mamba_multi_agent_data' / f'{safe}_{digest}.pkl'
        if source.exists() and not options.refresh_data_cache:
            result = original_data(options, logger)
        else:
            isolated = copy.copy(options)
            isolated.cache_dir = str(private_cache)
            result = original_data(isolated, logger)
        # Pin the actual vector destination in saved config metadata.
        if not options.skip_mamba and not options.item_vector_artifact:
            data = result[0]
            identity = options.mamba_model_id + "\n" + options.item_prompt_prefix + "\n" + "\n".join(data.item_texts)
            fingerprint = hashlib.sha1(identity.encode()).hexdigest()[:12]
            name = f"{safe}_{data.num_items}_{fingerprint}_prompt_tok{options.mamba_max_tokens}.pt"
            shared = Path(options.cache_dir) / 'mamba_multi_agent' / name
            options.item_vector_artifact = str(shared if shared.exists() else private_cache / 'mamba_multi_agent' / name)
        return result

    def load_text(texts, artifact, device, skip_mamba, cache_dir, **kwargs):
        if Path(artifact).exists():
            return original_text(texts, artifact, device, skip_mamba, cache_dir, **kwargs)
        destination = private_cache / 'mamba_multi_agent' / Path(artifact).name
        return original_text(texts, str(destination), device, skip_mamba, str(private_cache), **kwargs)

    train.load_recommendation_data_cached = load_data
    train.load_or_encode_text = load_text
    try:
        yield
    finally:
        train.load_recommendation_data_cached = original_data
        train.load_or_encode_text = original_text


def training_options(args, subset, mode, directory):
    dataset, validation_steps, cap, candidates, alpha, beta = SUBSETS[subset]
    options = [
        '--dataset', dataset, '--cache-dir', str(args.cache_dir),
        '--output-run-dir', str(directory), '--score-file', str(directory / 'scores.json'),
        '--specialist-epochs', str(args.epochs), '--joint-epochs', str(args.joint_epochs),
        '--batch-size', str(args.batch_size), '--eval-batch-size', str(args.eval_batch_size),
        '--validate-every-steps', str(validation_steps), '--monitor-metric', 'ndcg@10',
        '--early-stopping-patience', '6', '--lr-patience', '2',
        '--max-transitions', str(args.max_transitions or cap), '--candidates', str(candidates),
        '--short-window', str(args.short_window), '--preference-count', str(args.preference_count), '--max-history', '0',
        '--dim', '128', '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '.05',
        '--specialist-lr', str(args.specialist_lr), '--joint-lr', str(args.joint_lr),
        '--rl-coef', str(args.rl_coef), '--rl-topk', str(args.rl_topk), '--seed', str(args.seed),
        '--device', args.device, '--graph-device', args.device,
        '--mamba-model-id', args.mamba_model_id, '--mamba-max-tokens', '32',
        '--mamba-encode-batch-size', str(args.encode_batch_size),
        '--use-short', '1', '--use-preference', '1' if mode == 'full' else '0',
        '--use-coordinator' if mode == 'full' else '--no-use-coordinator',
        '--use-graph-embeddings', '--popularity-alpha', str(alpha), '--transition-beta', str(beta),
        '--validation-user-limit', '0', '--periodic-test-user-limit', '0',
        '--eval-length-threshold', '10', '--eval-length-basis', 'filtered_full_sequence',
        '--no-generate-reasons', '--no-save-model-weights', '--reason-count', '0',
        '--experiment-note', f'RQ5 {mode}; SW{args.short_window} P{args.preference_count}; run_4090 settings; validation-selected',
    ]
    if args.skip_mamba:
        options += ['--skip-mamba']
    return options


def train_model(args, subset, mode, directory):
    options = training_options(args, subset, mode, directory)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *options]
        with read_only_source_cache(args.output_dir / "cache"), capture_final_model() as captured:
            train.main()
        if not captured:
            raise RuntimeError('No final test model captured')
        return captured
    finally:
        sys.argv = old_argv


@torch.inference_mode()
def analyze_full(captured, args):
    model, data = captured['model'], captured['data']
    model.eval()
    users = sorted(data.test_target)
    if args.analysis_user_limit:
        users = users[:args.analysis_user_limit]
    valid_users = sorted(data.valid_target)
    if args.analysis_user_limit:
        valid_users = valid_users[:args.analysis_user_limit]
    device, batch = captured['device'], args.eval_batch_size
    with train.amp_context(device):
        graph = model.graph_item_vectors()
        items = model.project_all(graph)

    def batches(selected, split, description=None):
        for i in tqdm(range(0, len(selected), batch), desc=f'RQ5 {description or split}', unit='batch'):
            us = selected[i:i + batch]
            h, lengths, raw = train.evaluation_history_batch(data, us, split, 0, device)
            with train.amp_context(device):
                states = model.encode_states(h, lengths, graph)
            yield i, us, raw, states

    valid_entropy = []
    for _, _, raw, states in batches(valid_users, 'valid'):
        with train.amp_context(device):
            scores = model.short_agent.logits(states[0], items)
        valid_entropy.extend(entropy_from_scores(scores, raw).cpu().tolist())
    if not valid_entropy or not users:
        raise ValueError('RQ5 needs nonempty validation and test splits')
    thresholds = np.quantile(np.asarray(valid_entropy, dtype=np.float64), [1/3, 2/3])
    # Get every user's original rho before constructing subset-wide permutations.
    rho = np.empty(len(users), dtype=np.float32)
    for i, us, _, states in batches(users, 'test', 'test rho collection'):
        rho[i:i + len(us)] = states[7].float().cpu().numpy()
    rng = np.random.default_rng(args.seed + 501)
    permutations = np.stack([rng.permutation(len(users)) for _ in range(args.shuffle_repeats)])
    result = {'users': np.asarray(users), 'rho': rho, 'thresholds': thresholds,
              'permutations': permutations, 'validation_users': len(valid_users),
              'target': np.asarray([data.test_target[u] for u in users]),
              'full_rank': np.empty(len(users), dtype=np.int64),
              'entropy': np.empty(len(users)), 'weights': np.empty((len(users), 2)),
              'shuffled_rank': np.empty((args.shuffle_repeats, len(users)), dtype=np.int64),
              'shuffled_weight': np.empty((args.shuffle_repeats, len(users)))}
    for i, us, raw, states in batches(users, 'test', 'test ranking'):
        sl = slice(i, i + len(us))
        with train.amp_context(device):
            out = model.logits_from_states(states, items)
        result['entropy'][sl] = entropy_from_scores(out['short'], raw).cpu().numpy()
        result['weights'][sl] = states[3].float().cpu().numpy()
        result['full_rank'][sl] = masked_ranks(out['coordinator'], data, us, raw, captured)
        del out
        for repeat in range(args.shuffle_repeats):
            shuffled_rho = torch.tensor(rho[permutations[repeat, sl]], device=device)
            with train.amp_context(device):
                changed = coordinator_with_rho(model, states, shuffled_rho)
                out = model.logits_from_states(changed, items)
            result['shuffled_rank'][repeat, sl] = masked_ranks(out['coordinator'], data, us, raw, captured)
            result['shuffled_weight'][repeat, sl] = changed[3][:, 1].float().cpu().numpy()
            del out, changed
    result['groups'] = assign_groups(result['entropy'], thresholds)
    return result


@torch.inference_mode()
def analyze_short(captured, users, args):
    model, data = captured['model'], captured['data']
    model.eval()
    if any(u not in data.test_target for u in users):
        raise ValueError('Full and short-only evaluation populations differ')
    ranks = np.empty(len(users), dtype=np.int64)
    with train.amp_context(args.device):
        graph = model.graph_item_vectors()
        items = model.project_all(graph)
    for i in tqdm(range(0, len(users), args.eval_batch_size), desc='RQ5 short-only'):
        us = [int(u) for u in users[i:i + args.eval_batch_size]]
        h, lengths, raw = train.evaluation_history_batch(data, us, 'test', 0, args.device)
        with train.amp_context(args.device):
            output = model.full_catalog_scores(h, lengths, items, graph)
        ranks[i:i + len(us)] = masked_ranks(output['coordinator'], data, us, raw, captured)
    return ranks


def build_summary(result, args):
    full = rank_metrics(result['full_rank'])
    short = rank_metrics(result['short_rank'])
    shuffled_each = rank_metrics(result['shuffled_rank'])
    shuffled = shuffled_each.mean(axis=0)
    summary, bootstraps = {}, {}
    for index, group in enumerate(GROUPS):
        mask = result['groups'] == index
        n = int(mask.sum())
        weights = result['weights'][mask, 1]
        entry = {'users': n, 'metrics': {}, 'gains': {}, 'preference_weight': None}
        if n:
            entry['preference_weight'] = {'mean': float(weights.mean()), 'median': float(np.median(weights)),
                                         'q25': float(np.quantile(weights, .25)), 'q75': float(np.quantile(weights, .75))}
        for name, values in [('full', full), ('short_only', short), ('rho_shuffled_mean', shuffled)]:
            entry['metrics'][name] = dict(zip(METRICS, values[mask].mean(axis=0).tolist())) if n else None
        entry['rho_shuffle_repeat_metrics'] = [dict(zip(METRICS, x[mask].mean(axis=0).tolist())) if n else None
                                               for x in shuffled_each]
        for control, values in [('short_only', short), ('rho_shuffled_mean', shuffled)]:
            entry['gains'][control] = {}
            for k, metric in enumerate(METRICS):
                difference = (full - values)[mask, k]
                samples = bootstrap_means(difference, args.bootstrap, np.random.default_rng(args.seed + index * 100 + k))
                bootstraps[group, control, metric] = samples
                entry['gains'][control][metric] = {'mean': float(difference.mean()) if n else None,
                                                  'ci95': interval(samples)}
        summary[group] = entry
    contrasts = {}
    for control in ('short_only', 'rho_shuffled_mean'):
        contrasts[control] = {}
        for metric in METRICS:
            high, low = (bootstraps[g, control, metric] for g in ('high', 'low'))
            delta = (summary['high']['gains'][control][metric]['mean'] - summary['low']['gains'][control][metric]['mean']) if high is not None and low is not None else None
            contrasts[control][metric] = {'high_minus_low_gain': delta,
                                         'ci95': interval(high - low) if delta is not None else None}
    return summary, contrasts


def write_report(path, subset, result, args):
    summary, contrasts = build_summary(result, args)
    full, short = rank_metrics(result['full_rank']), rank_metrics(result['short_rank'])
    shuffled = rank_metrics(result['shuffled_rank'])
    metadata = {
        'dataset': SUBSETS[subset][0], 'subset': subset, 'seed': args.seed,
        'configuration': {'short_window': args.short_window, 'preference_count': args.preference_count, 'max_history': 0,
                          'text_encoder': args.mamba_model_id, 'epochs': args.epochs, 'joint_epochs': args.joint_epochs},
        'smoke_test_random_features': args.skip_mamba,
        'analysis_user_limit': args.analysis_user_limit,
        'id_space': 'Integer indices from the filtered dataset; one held-out test target per user.',
        'uncertainty_definition': 'Normalized entropy of full RIBS short-head softmax (temperature=1), excluding observed items, without priors or target-dependent masking.',
        'thresholds': {'low_medium': float(result['thresholds'][0]), 'medium_high': float(result['thresholds'][1]),
                       'source': 'validation', 'validation_users': result['validation_users'],
                       'rules': 'low <= lower; medium > lower and <= upper; high > upper; ties kept together'},
        'shuffle': {'repeats': args.shuffle_repeats, 'seed': args.seed + 501,
                    'scope': 'entire analyzed test subset, before batching',
                    'intervention': 'Only coordinator rho context is shuffled; states and predicted preference certainty are preserved.'},
        'inference': 'Paired user bootstrap; shuffle mean averaged within user before bootstrap. Shuffle-repeat variability reported separately; no training-seed uncertainty estimate.',
        'bootstrap_repeats': args.bootstrap, 'summary': summary, 'high_minus_low': contrasts,
    }
    # Stream records: do not keep hundreds of thousands of Python dictionaries in RAM.
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False)[:-1] + ', "groups": {\n')
        for group_index, group in enumerate(GROUPS):
            if group_index:
                stream.write(',\n')
            stream.write(json.dumps(group) + ': [\n')
            first = True
            for i in np.flatnonzero(result['groups'] == group_index):
                record = {'user_id': int(result['users'][i]), 'target_item_id': int(result['target'][i]),
                          'short_uncertainty': float(result['entropy'][i]),
                          'preference_weight': float(result['weights'][i, 1]),
                          'short_weight': float(result['weights'][i, 0]),
                          'transition_probability': float(result['rho'][i]),
                          **dict(zip(METRICS, full[i].tolist())),
                          'short_only': dict(zip(METRICS, short[i].tolist())),
                          'rho_shuffled': {'preference_weights': result['shuffled_weight'][:, i].tolist(),
                              'transition_probabilities': result['rho'][result['permutations'][:, i]].tolist(),
                              'metrics_per_repeat': [dict(zip(METRICS, x[i].tolist())) for x in shuffled]}}
                if not first:
                    stream.write(',\n')
                stream.write(json.dumps(record, allow_nan=False))
                first = False
            stream.write('\n]')
        stream.write('\n}}\n')
    temporary.replace(path)
    path.with_name('rq5_summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--subset', choices=SUBSETS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=root.parents[1] / 'cache')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--short-window', type=int, choices=[4], default=4)
    parser.add_argument('--preference-count', type=int, choices=[32], default=32)
    parser.add_argument('--mamba-model-id', default='state-spaces/mamba-130m-hf')
    parser.add_argument('--joint-epochs', type=int, default=0)
    parser.add_argument('--specialist-lr', type=float, default=1e-4)
    parser.add_argument('--joint-lr', type=float, default=5e-5)
    parser.add_argument('--rl-coef', type=float, default=.1)
    parser.add_argument('--rl-topk', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--encode-batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=25252)
    parser.add_argument('--shuffle-repeats', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--analysis-user-limit', type=int, default=0, help='Debug only; 0 evaluates all users.')
    parser.add_argument('--max-transitions', type=int, default=0, help='Debug training cap; 0 uses dataset defaults.')
    parser.add_argument('--skip-mamba', action='store_true', help='Smoke test only: random text features.')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.encode_batch_size,
           args.shuffle_repeats, args.bootstrap, args.short_window, args.preference_count, args.rl_topk) < 1 or min(args.analysis_user_limit, args.max_transitions, args.joint_epochs) < 0:
        parser.error('Counts must be positive; debug limits/joint epochs must be nonnegative')
    if args.preference_count < 2 or min(args.specialist_lr, args.joint_lr) <= 0 or args.rl_coef < 0:
        parser.error('Preference count must be >=2; learning rates positive; RL coefficient nonnegative')
    # A fresh output directory prevents overwriting an earlier RQ5 or other run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'rq5_config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    captured = train_model(args, args.subset, 'full', args.output_dir / 'full')
    result = analyze_full(captured, args)
    np.savez_compressed(args.output_dir / 'full_analysis.npz', **result)
    captured.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    captured = train_model(args, args.subset, 'short_only', args.output_dir / 'short_only')
    result['short_rank'] = analyze_short(captured, result['users'], args)
    assert np.array_equal(result['target'], [captured['data'].test_target[int(u)] for u in result['users']])
    np.savez_compressed(args.output_dir / 'rq5_arrays.npz', **result)
    write_report(args.output_dir / 'rq5.json', args.subset, result, args)
    print(f'RQ5 complete: {args.output_dir / "rq5.json"}')


if __name__ == '__main__':
    main()
