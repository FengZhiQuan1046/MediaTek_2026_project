"""Recover original timestamps and summarize 30-day groups for preliminary figures.

No downloads, retraining or changes to source datasets/figures. Large Arrow inputs
remain memory-mapped in the workspace cache; only compact summaries enter the repo.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

DAY_MS = 86_400_000


def time_groups(timestamps, gap_days=30):
    """Consecutive-gap grouping: equality stays together; missing dates excluded."""
    dated = sorted(t for t in timestamps if t is not None)
    groups = []
    for t in dated:
        if not groups or t - groups[-1][-1] > gap_days * DAY_MS:
            groups.append([t])
        else:
            groups[-1].append(t)
    return groups


def distribution(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'count': 0, 'mean': None, 'median': None, 'p25': None,
                'p75': None, 'min': None, 'max': None}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p25': float(np.quantile(a, .25)), 'p75': float(np.quantile(a, .75)),
            'min': float(a.min()), 'max': float(a.max())}


def summarize(histories, users):
    sizes, spans, per_user, group_counts = [], [], [], []
    ordinal = defaultdict(lambda: {'sizes': [], 'spans': []})
    missing = 0
    users = sorted(set(users))
    for user in users:
        h = histories[user]
        missing += sum(t is None for t in h)
        groups = time_groups(h)
        group_counts.append(len(groups))
        if groups:
            per_user.append(sum(map(len, groups)) / len(groups))
        for index, group in enumerate(reversed(groups), 1):
            span = (group[-1] - group[0]) / DAY_MS
            sizes.append(len(group)); spans.append(span)
            ordinal[index]['sizes'].append(len(group))
            ordinal[index]['spans'].append(span)
    return {'users': len(users), 'groups': len(sizes), 'dated_interactions': sum(sizes),
            'missing_timestamp_interactions': missing,
            'mean_interactions_per_group': distribution(sizes)['mean'],
            'mean_group_span_days': distribution(spans)['mean'],
            'group_length_interactions': distribution(sizes),
            'group_span_days': distribution(spans),
            'groups_per_user': distribution(group_counts),
            'user_weighted_mean_interactions_per_group': distribution(per_user)['mean'],
            'by_group_from_most_recent': [
                {'group': index, 'contributing_users': len(values['sizes']),
                 'mean_interactions': distribution(values['sizes'])['mean'],
                 'median_interactions': distribution(values['sizes'])['median'],
                 'mean_span_days': distribution(values['spans'])['mean']}
                for index, values in sorted(ordinal.items())]}


def csv_users(path, split='test'):
    with path.open(newline='') as stream:
        return {int(row['user']) for row in csv.DictReader(stream) if row['split'] == split}


def recover(source, cache_dir, selected_users, data):
    import pyarrow as pa
    import pyarrow.compute as pc
    category = source['category']
    files = sorted((cache_dir/'McAuley-Lab___amazon-reviews-2023'/
                    ('raw_review_' + category)).rglob('*.arrow'))
    assert files, f'No local raw Arrow cache for {category}'
    assert len({p.parent for p in files}) == 1, 'Ambiguous raw cache revisions'
    print(category, 'loading', len(files), 'raw Arrow shards', flush=True)
    tables = []
    for path in files:
        columns = ['user_id', 'parent_asin', 'timestamp']
        with pa.memory_map(str(path), 'r') as stream:
            schema = pa.ipc.open_stream(stream).schema
        indices = [schema.get_field_index(name) for name in columns]
        assert all(index >= 0 for index in indices), str(path)
        with pa.memory_map(str(path), 'r') as stream:
            options = pa.ipc.IpcReadOptions(included_fields=indices)
            tables.append(pa.ipc.open_stream(stream, options=options).read_all().select(columns))
    raw = pa.concat_tables(tables)
    raw = raw.filter(pc.and_(pc.and_(pc.is_valid(raw['user_id']),
                pc.not_equal(raw['user_id'], '')), pc.and_(pc.is_valid(raw['parent_asin']),
                pc.not_equal(raw['parent_asin'], ''))))
    print(category, 'counting raw user/item degrees for', raw.num_rows, 'rows', flush=True)
    uc = raw.select(['user_id']).group_by('user_id').aggregate([('user_id', 'count')])
    ic = raw.select(['parent_asin']).group_by('parent_asin').aggregate([('parent_asin', 'count')])
    good_users = uc['user_id'].filter(pc.greater_equal(uc['user_id_count'], 5))
    good_items = ic['parent_asin'].filter(pc.greater_equal(ic['parent_asin_count'], 5))
    # Match src.data.build_data: count once, filter simultaneously, retain
    # surviving users even when fewer than five interactions remain.
    filtered = raw.filter(pc.and_(pc.is_in(raw['user_id'], value_set=good_users),
                                 pc.is_in(raw['parent_asin'], value_set=good_items)))
    all_users = sorted(pc.unique(filtered['user_id']).to_pylist())
    assert len(all_users) == data.num_users, (category, len(all_users), data.num_users)
    numeric_of = {all_users[u]: u for u in selected_users}
    small = filtered.filter(pc.is_in(filtered['user_id'], value_set=pa.array(list(numeric_of))))
    per_user = defaultdict(list)
    rows = small.to_pydict()
    for user, item, ts in zip(rows['user_id'], rows['parent_asin'], rows['timestamp']):
        per_user[numeric_of[user]].append((item, ts))
    mapping, reverse_mapping, histories = {}, {}, {}
    for user in sorted(selected_users):
        # Sorting is stable, including timestamp ties, matching original input.
        seq = sorted(per_user[user], key=lambda pair: pair[1] or 0)
        ids = list(data.train_by_user[user])
        if user in data.valid_target: ids.append(data.valid_target[user])
        if user in data.test_target: ids.append(data.test_target[user])
        assert len(seq) == len(ids), (category, user, len(seq), len(ids))
        for item_id, (original, _) in zip(ids, seq):
            assert mapping.setdefault(item_id, original) == original, (category, user, 'item order mismatch')
            assert reverse_mapping.setdefault(original, item_id) == item_id, (category, user, 'ID mismatch')
        histories[user] = [pair[1] for pair in seq]
    print(category, 'verified user counts and complete cached sequences for', len(histories), 'users', flush=True)
    return histories, {'raw_rows': raw.num_rows, 'filtered_rows': filtered.num_rows,
        'filtered_users': len(all_users), 'verified_user_sequences': len(histories),
        'verified_distinct_item_ids': len(mapping), 'raw_arrow_files': [str(p) for p in files]}


def run_subset(source, args):
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    folder = Path(source['source'])
    manifest = json.loads((folder/'manifest.json').read_text())
    diagnostic = next(d for d in manifest['datasets'] if d['dataset'] == source['dataset'])
    cache = Path(diagnostic['cache'])
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    assert digest == diagnostic['cache_sha256'], 'Preliminary cache changed'
    with cache.open('rb') as stream: data = pickle.load(stream)
    settings = manifest['settings']
    sampled = np.array(sorted(set(data.valid_target) & set(data.test_target)), dtype=np.int64)
    if settings['max_users'] and settings['max_users'] < len(sampled):
        sampled = np.sort(np.random.default_rng(settings['seed']).choice(
            sampled, settings['max_users'], replace=False))
    influence = csv_users(Path(diagnostic['model_item_influence_source']))
    coverage = csv_users(folder/'information_coverage_per_user.csv')
    requested = set(map(int, sampled)) | influence | coverage
    restored, validation = recover(source, args.cache_dir, requested, data)
    full = {u: restored[u][:len(data.train_by_user[u])+1] for u in requested}
    plotted = {u: h[-settings['max_lag']:] for u, h in full.items()}
    influence_rows_checked = 0
    with Path(diagnostic['model_item_influence_source']).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] != 'test': continue
            user = int(row['user'])
            ids = (list(data.train_by_user[user]) + [data.valid_target[user]])[-settings['max_lag']:]
            assert len(ids) == int(row['history_length'])
            assert ids[-int(row['lag'])] == int(row['item'])
            assert data.test_target[user] == int(row['target'])
            influence_rows_checked += 1
    with (folder/'information_coverage_per_user.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == 'test':
                assert len(plotted[int(row['user'])]) == int(row['history_length'])
    vectorizer = HashingVectorizer(n_features=1024, ngram_range=(1, 2),
                                   alternate_sign=False, norm='l2', stop_words='english')
    usable = np.asarray(vectorizer.transform(data.item_texts).getnnz(axis=1)).reshape(-1) > 0
    text_users = set()
    for user in map(int, sampled):
        h = (list(data.train_by_user[user]) + [data.valid_target[user]])[-settings['max_lag']:]
        if h and usable[data.test_target[user]] and any(usable[i] for i in h):text_users.add(user)
    fixed = {u for u in text_users if len(plotted[u]) >= 16}
    # Exact original cohort counts provide an independent alignment check.
    with (folder/'alignment_summary.csv').open(newline='') as stream:
        checks = [r for r in csv.DictReader(stream) if r['split']=='test' and
                  r['cohort']=='all_available' and r['lag_start']=='1' and r['lag_end']=='1']
    assert checks and all(int(r['n_users']) == len(text_users) for r in checks), 'Figure 1/2 cohort mismatch'
    assert len(coverage) == diagnostic['information_coverage_users']['test'], 'Figure 4 cohort mismatch'
    report = {
        'subset': source['category'], 'dataset': source['dataset'],
        'split': 'test',
        'grouping': {'rule': 'new group iff consecutive observed timestamp gap > 30 days',
                     'gap_days': 30, 'exactly_30_days_stays_in_group': True,
                     'timestamp_unit': 'Unix milliseconds (UTC)',
                     'group_1': 'most recent group', 'singleton_span_days': 0,
                     'missing_timestamps': 'excluded from temporal grouping; counted separately',
                     'rule_evidence': 'preliminary_grouped/analyze.py (CPython 3.12 test_histories bytecode): Jan1,Jan1,Jan2,Feb1 form size4 group; Mar3 is separate'},
        'history': {'construction': 'train + observed validation item; test target excluded',
                    'plotted_max_interactions': settings['max_lag'],
                    'group_after_history_truncation': True,
                    'duplicates': 'retained exactly as original preliminary preprocessing'},
        'mean_interactions_per_group': summarize(plotted, text_users)['mean_interactions_per_group'],
        'mean_group_span_days': summarize(plotted, text_users)['mean_group_span_days'],
        'figure_cohorts': {
            'figure1_alignment_and_figure2_preference_all_available': summarize(plotted, text_users),
            'figure1_alignment_and_figure2_preference_fixed_K16': summarize(plotted, fixed),
            'figure3_trained_model_influence': summarize(plotted, influence),
            'figure4_information_coverage': summarize(plotted, coverage)},
        'full_untruncated_observed_history_reference': summarize(full, text_users),
        'provenance': {'figure_sources': str(folder), 'preliminary_cache': str(cache),
                       'preliminary_cache_sha256': digest, 'timestamp_recovery': validation,
                       'seed': settings['seed'], 'sampled_users': len(sampled),
                       'figure3_verified_lag_item_target_rows': influence_rows_checked,
                       'figure4_history_lengths_verified': True,
                       'no_training_or_figures_modified': True},
        'interpretation': 'Temporal groups supplement the existing four figures; they are not the original semantic KMeans preference clusters. A 30-day consecutive-gap group can span more than 30 days.'}
    destination = args.output / (source['category'] + '.json')
    args.output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(source['category'], 'mean interactions/group=', report['mean_interactions_per_group'],
          'mean span days=', report['mean_group_span_days'], '->', destination, flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, default=root.parent/'cache')
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'outputs/amazon_group_lengths')
    args = p.parse_args()
    import pyarrow as pa
    pa.set_cpu_count(4)
    sys.path.insert(0, str(root/'ver4'))
    sources = json.loads((root/'preliminary/aggregated/sources.json').read_text())['sources']
    for source, category in zip(sources, ['All_Beauty','Baby_Products','Sports_and_Outdoors','Toys_and_Games']):
        source = dict(source, category=category)
        run_subset(source, args)


if __name__ == '__main__':main()
