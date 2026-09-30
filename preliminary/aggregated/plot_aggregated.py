#!/usr/bin/env python3
"""Replot existing preliminary CSV summaries; no training or new estimates.

Requires matplotlib and numpy. Run from any working directory:
    python /path/to/preliminary/aggregated/plot_aggregated.py
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/preliminary-aggregated-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = [
    ('amazon-all-beauty', 'amazon-all-beauty', 'All Beauty', '#0072B2', 'o', '-'),
    ('amazon_Baby_Products', 'amazon:Baby_Products', 'Baby Products', '#D55E00', 's', '--'),
    ('amazon-sports-and-outdoors', 'amazon-sports-and-outdoors', 'Sports & Outdoors', '#009E73', '^', '-.'),
    ('amazon-toys-and-games', 'amazon-toys-and-games', 'Toys & Games', '#CC79A7', 'D', ':'),
]
FILES = ['alignment_summary.csv', 'model_item_influence_summary.csv', 'information_coverage.csv']
SETTINGS = ['max_users', 'max_lag', 'short_window', 'random_matches', 'reference_items',
            'normalization_pairs', 'cluster_items', 'clusters', 'bootstrap', 'seed']


def load_sources(root, pinned=None):
    tables = {name: [] for name in FILES}
    provenance = []
    reference = None
    for folder, dataset, *_ in DATASETS:
        if pinned:
            candidates = [root / folder / pinned[folder]]
        else:
            candidates = sorted((root / folder).glob('run_*'), reverse=True)
        selected = None
        for run in candidates:
            manifest_path = run / 'manifest.json'
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('status') == 'complete' and all((run / f).exists() for f in FILES):
                selected = run
                break
        if selected is None:
            raise ValueError(f'No complete analysis with all summary CSVs for {folder}')
        signature = (manifest['method'], {k: manifest['settings'][k] for k in SETTINGS})
        if reference is not None and signature != reference:
            raise ValueError(f'Incompatible analysis method/settings: {selected}')
        reference = signature
        hashes = {}
        for filename in FILES:
            path = selected / filename
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            if not rows or any(row['dataset'] != dataset for row in rows):
                raise ValueError(f'Unexpected or missing dataset rows in {path}')
            if filename == 'alignment_summary.csv' and not any(
                    row['metric'] == 'history_normalized_cluster_match' for row in rows):
                raise ValueError(f'History-normalized preference statistic missing from {path}')
            tables[filename].extend(rows)
        provenance.append({'folder': folder, 'dataset': dataset, 'run': selected.name,
                           'source': str(selected.resolve()), 'sha256': hashes,
                           'method': manifest['method'], 'settings': signature[1],
                           'limitations': manifest.get('limitations', [])})
    return tables, provenance


def style(font_path=None):
    if font_path:
        font_path = Path(font_path).expanduser().resolve()
        if not font_path.is_file():
            raise FileNotFoundError(f'Times New Roman font not found: {font_path}')
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        if family != 'Times New Roman':
            raise ValueError(f'{font_path} is {family!r}, not Times New Roman')
    else:
        matches = [f for f in font_manager.fontManager.ttflist if f.name == 'Times New Roman']
        family = 'Times New Roman' if matches else 'Liberation Serif'
        if not matches:
            print('WARNING: Times New Roman is unavailable; using Liberation Serif preview. '
                  'Supply --font-path /path/to/times.ttf for final submission.')
    plt.rcParams.update({
        'font.family': family,
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'axes.edgecolor': '#666666',
        'grid.color': '#D8DCE1', 'grid.linewidth': .5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
    return family


def legend(fig):
    handles = [Line2D([], [], color=color, marker=marker, linestyle=ls,
                      lw=1.6, markersize=4, label=name)
               for _, _, name, color, marker, ls in DATASETS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .998),
               ncol=4, frameon=False, columnspacing=1.6, handlelength=2.5)


def decorate(ax, title, ylabel, xlabel, zero=True):
    ax.set_title(title, loc='left', pad=9, fontweight='semibold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', alpha=.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3, width=.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if zero:
        ax.axhline(0, color='#777777', lw=.75, ls='--', zorder=1)


def draw(ax, tables, kind, min_users, cohort='all_available', full=False, preference_count=None):
    missing = []
    for _, dataset, name, color, marker, ls in DATASETS:
        if kind in ('alignment', 'preference'):
            metric = 'excess_cosine_z' if kind == 'alignment' else 'history_normalized_cluster_match'
            rows = [r for r in tables[FILES[0]] if r['dataset'] == dataset and r['split'] == 'test'
                    and r['cohort'] == cohort and r['metric'] == metric
                    and int(r['n_users']) >= min_users]
            rows.sort(key=lambda r: float(r['lag_start']))
            x = [(float(r['lag_start']) + float(r['lag_end'])) / 2 for r in rows]
            y, low, high = 'mean', 'ci_low', 'ci_high'
        elif kind == 'influence':
            rows = [r for r in tables[FILES[1]] if r['dataset'] == dataset and r['split'] == 'test'
                    and (full or int(r['n_users']) >= min_users)]
            rows.sort(key=lambda r: int(r['lag']))
            x = [int(r['lag']) for r in rows]
            y, low, high = 'mean_margin_influence', 'ci_low', 'ci_high'
        else:
            rows = [r for r in tables[FILES[2]] if r['dataset'] == dataset and r['split'] == 'test']
            rows.sort(key=lambda r: int(r['recent_items']))
            x = [int(r['recent_items']) for r in rows]
            y, low, high = 'mean_percent', 'q25_percent', 'q75_percent'
        if not rows:
            missing.append(name)
            continue
        values = np.array([[float(r[k]) for k in (y, low, high)] for r in rows])
        if (not np.isfinite(values[:, 0]).all()
                or (not full and not np.isfinite(values).all())
                or np.any(values[:, 1] > values[:, 2])):
            raise ValueError(f'Invalid plotted statistics: {dataset}, {kind}, {cohort}')
        ax.fill_between(x, values[:, 1], values[:, 2], color=color,
                        alpha=.08 if kind == 'coverage' else .11, linewidth=0, zorder=1)
        ax.plot(x, values[:, 0], color=color, ls=ls, lw=1.6, marker=marker,
                markersize=3.5, markeredgewidth=.65, markerfacecolor='white',
                markevery=max(1, len(x) // 9), zorder=3)
    ax.set_xlim(left=.5)
    if kind in ('alignment', 'preference'):
        ax.set_xticks([1, 2, 3.5, 6.5, 12.5, 24.5][:6 if cohort == 'all_available' else 5])
        ax.set_xticklabels(['1', '2', '3–4', '5–8', '9–16', '17–32'][:6 if cohort == 'all_available' else 5])
    if kind == 'preference':
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    if kind == 'coverage':
        ax.set_ylim(0, 103)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.axhline(80, color='#777777', lw=.65, ls='--', zorder=1)
    if missing:
        ax.text(.98, .97, '\n'.join(f'{name}: n < {min_users}' for name in missing),
                ha='right', va='top', transform=ax.transAxes, fontsize=7, color='#666666')
    return missing


LABELS = {
    'alignment': ('Semantic alignment', 'Excess cosine similarity (z)', 'Interaction lag (binned; 1 = most recent)'),
    'preference': ('Preference alignment', 'Excess match vs. history chance', 'Interaction lag (binned; 1 = most recent)'),
    'influence': ('History-item influence', 'Next-item margin change', 'Interaction lag (1 = most recent)'),
    'coverage': ('Cumulative signal coverage', 'Positive semantic signal covered', 'Number of most recent items'),
}


def save(fig, out, stem, formats, dpi):
    for ext in formats:
        fig.savefig(out / f'{stem}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, default=HERE.parent / 'outputs' / 'analysis')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--sources', type=Path, help='Pin runs using a previous sources.json')
    parser.add_argument('--min-users', type=int, default=100)
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf', 'png', 'svg'])
    parser.add_argument('--dpi', type=int, default=400)
    parser.add_argument('--font-path', type=Path, help='Times New Roman TTF/OTF file for final figures')
    args = parser.parse_args()
    if args.min_users < 1 or args.dpi < 1:
        parser.error('--min-users and --dpi must be positive')
    pinned = None
    if args.sources:
        pinned = {x['folder']: x['run'] for x in json.loads(args.sources.read_text())['sources']}
    tables, provenance = load_sources(args.analysis_root, pinned)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    font_family = style(args.font_path)
    n = args.min_users
    preference_count = provenance[0]['settings']['clusters']
    ci_note = f'Test split · Shading: 95% user-bootstrap CI · Points require n ≥ {n} users'
    # Three panel figure sized for a typical two-column conference page (7.15 in).
    # The coverage panel focuses on the first 20 items; the standalone figure retains 1–100.
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    specs = [
        ('alignment', '(a) Semantic alignment', 'Excess cosine (z)', 'Lag bin'),
        ('influence', '(c) Item influence', 'Margin change', 'Lag'),
        ('coverage', '(d) Signal coverage', 'Signal covered (%)', 'Recent items'),
    ]
    for ax, (kind, title, ylabel, xlabel) in zip(axes, specs):
        decorate(ax, title, ylabel, xlabel, kind != 'coverage')
        draw(ax, tables, kind, n)
        if kind == 'coverage':
            ax.set_xlim(.5, 20.5)
            ax.set_xticks([1, 5, 10, 15, 20])
        if kind == 'alignment':
            ax.set_xticks([1, 3.5, 12.5, 24.5], labels=['1', '3–4', '9–16', '17–32'])
        if kind == 'influence':
            ax.set_xticks([1, 5, 10, 15])
        ax.title.set_fontsize(8.5)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.tick_params(labelsize=7)
    legend(fig)
    fig.subplots_adjust(left=.085, right=.985, top=.78, bottom=.21, wspace=.20)
    save(fig, out, 'figure0_acd_row', args.formats, args.dpi)

    # Wider four-panel alternative, with a divider before the preference panel.
    fig, axes = plt.subplots(1, 4, figsize=(8.80, 2.50))
    four_specs = [
        ('alignment', '(a) Semantic', 'Excess cosine (z)', 'Lag bin'),
        ('influence', '(b) Influence', 'Margin change', 'Lag'),
        ('coverage', '(c) Coverage', 'Signal (%)', 'Recent items'),
        ('preference', '(d) Preference match', 'Excess match (pp)', 'Lag bin'),
    ]
    for ax, (kind, title, ylabel, xlabel) in zip(axes, four_specs):
        decorate(ax, title, ylabel, xlabel, kind != 'coverage')
        draw(ax, tables, kind, n, preference_count=preference_count)
        if kind in ('alignment', 'preference'):
            ax.set_xticks([1, 12.5, 24.5], labels=['1', '9–16', '17–32'])
        elif kind == 'influence':
            ax.set_xticks([1, 8, 16])
        else:
            ax.set_xlim(.5, 20.5)
            ax.set_xticks([1, 10, 20])
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
        ax.tick_params(labelsize=6.5)
    legend(fig)
    fig.subplots_adjust(left=.075, right=.955, top=.77, bottom=.21, wspace=.28)
    # Move the final panel right while preserving its width. The larger
    # canvas supplies the extra space, so all four plotting areas stay equal.
    right_panel = axes[3].get_position()
    extra_gap = .029
    axes[3].set_position([right_panel.x0 + extra_gap, right_panel.y0,
                          right_panel.width, right_panel.height])
    gap_left = axes[2].get_position().x1
    gap_right = axes[3].get_position().x0
    divider_x = gap_left + .30 * (gap_right - gap_left)
    fig.add_artist(Line2D([divider_x, divider_x], [.15, .82],
                          transform=fig.transFigure, color='#4B5563', lw=1.55))
    save(fig, out, 'figure0_abcd_row', args.formats, args.dpi)

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    decorate(ax, '(b) Preference alignment', 'Excess match vs. history chance', 'Lag bin')
    draw(ax, tables, 'preference', n, preference_count=preference_count)
    ax.set_xticks([1, 3.5, 12.5, 24.5], labels=['1', '3–4', '9–16', '17–32'])
    legend(fig)
    fig.subplots_adjust(left=.20, right=.97, top=.77, bottom=.21)
    save(fig, out, 'figure0_b_preference', args.formats, args.dpi)
    missing = {}
    for number, kind in enumerate(LABELS, 1):
        title, ylabel, xlabel = LABELS[kind]
        if number <= 2:
            fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=(kind != 'preference'))
            for letter, ax, cohort, subtitle in zip('ab', axes, ['all_available', 'fixed_K16'],
                                                   ['Available-history cohort', 'Fixed cohort: history ≥ 16']):
                decorate(ax, f'({letter}) {subtitle}', ylabel, xlabel)
                omitted = draw(ax, tables, kind, n, cohort, preference_count=preference_count)
                if omitted:
                    missing[f'{kind}/{cohort}'] = omitted
            axes[1].set_ylabel('')
            note = ci_note
            fig.subplots_adjust(left=.08, right=.985, top=.81, bottom=.23, wspace=.18)
        else:
            fig, ax = plt.subplots(figsize=(8.1, 4.4))
            decorate(ax, title, ylabel, xlabel, kind != 'coverage')
            draw(ax, tables, kind, n)
            note = ci_note if kind == 'influence' else ('Test split · Lines: user means; shading: user interquartile range\n'
                   'Users with zero positive signal excluded; reference line: 80%')
            fig.subplots_adjust(left=.12, right=.98, top=.81, bottom=.23)
        legend(fig)
        fig.text(.5, .035, note, ha='center', fontsize=7.5, color='#555555')
        save(fig, out, f'figure{number}_{kind}', args.formats, args.dpi)
    fig, ax = plt.subplots(figsize=(8.1, 4.4))
    decorate(ax, 'History-item influence — all observed positions', LABELS['influence'][1], LABELS['influence'][2])
    draw(ax, tables, 'influence', n, full=True)
    legend(fig)
    fig.text(.5, .025, 'Test split · 95% user-bootstrap CI where available · All points, including n < '
             f'{n}\nDistant positions may contain very few users; see plotted_data.csv for sample sizes.',
             ha='center', fontsize=7.5, color='#555555')
    fig.subplots_adjust(left=.12, right=.98, top=.81, bottom=.24)
    save(fig, out, 'figureS1_influence_all_positions', args.formats, args.dpi)
    # Retain exact source values, including points omitted from the main figures.
    records = []
    for filename, rows in tables.items():
        for row in rows:
            if row['split'] == 'test':
                records.append({'source_table': filename, **row})
    keys = list(dict.fromkeys(k for row in records for k in row))
    with (out / 'plotted_data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    (out / 'sources.json').write_text(json.dumps({
        'sources': provenance, 'split': 'test', 'min_users_main': n,
        'missing_main_curves': missing, 'formats': args.formats, 'png_dpi': args.dpi,
        'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
        'font_family': font_family, 'font_path': str(args.font_path) if args.font_path else None,
        'paper_coverage_x_range': [1, 20],
        'aggregation': 'Overlay subset estimates; no pooling, smoothing, or recomputation.',
        'preference_plot_metric': 'history_normalized_cluster_match',
        'preference_reference': 'per-user 1/K if target cluster occurs in usable history, else 0',
    }, indent=2) + '\n')
    print(f'Wrote 8 figures in {", ".join(args.formats)} to {out}; font: {font_family}')
    print(f'Omitted curves below the sample threshold: {missing}')


if __name__ == '__main__':
    main()
