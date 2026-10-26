"""Five preliminary figures aggregated by 30-day temporal periods on one isolated GPU."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

import amazon_group_lengths as lengths


def group_lags(timestamps):
    """Map each original lag to a recent-first temporal group without losing ties."""
    result = {}
    group = 1
    previous = None
    for lag, ts in enumerate(reversed(timestamps), 1):
        if ts is None:
            raise ValueError('Missing timestamp: cannot assign exact original lag')
        if previous is not None and previous - ts > 30 * lengths.DAY_MS:
            group += 1
        result[lag] = group
        previous = ts
    return result


def read_rows(path):
    with path.open(newline='') as stream:
        return [r for r in csv.DictReader(stream) if r['split']=='test']


def gpu_summary(values, seed, repetitions, device):
    a = torch.tensor(values, dtype=torch.float64, device=device)
    if not len(a):return None
    mean = float(a.mean())
    result = {'mean': mean, 'n_users': len(values), 'ci_low': None, 'ci_high': None}
    if len(a)>1 and repetitions:
        generator = torch.Generator(device=device).manual_seed(seed)
        samples = []
        for start in range(0, repetitions, 128):
            indices = torch.randint(len(a), (min(128,repetitions-start),len(a)),
                                    generator=generator, device=device)
            samples.append(a[indices].mean(1))
        q = torch.quantile(torch.cat(samples), torch.tensor([.025,.975],dtype=a.dtype,device=device))
        result.update(ci_low=float(q[0]),ci_high=float(q[1]))
    return result


def per_group_stats(values, seed, repetitions, device):
    """Each row of values[group] already represents one independent user mean."""
    return [dict(group=g, **gpu_summary(v, seed+g, repetitions, device))
            for g,v in sorted(values.items())]


def calculate(source, args, root):
    category = source['category']
    print(category,'restore histories and verify original signals',flush=True)
    folder = Path(source['source'])
    manifest = json.loads((folder/'manifest.json').read_text())
    diagnostic = next(d for d in manifest['datasets'] if d['dataset']==source['dataset'])
    cache = Path(diagnostic['cache'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==diagnostic['cache_sha256']
    with cache.open('rb') as stream:data=pickle.load(stream)
    safe=source['dataset'].replace(':','_')
    positions_path=folder/safe/'alignment_per_position.csv'
    positions=read_rows(positions_path)
    influence=read_rows(Path(diagnostic['model_item_influence_source']))
    selected={int(r['user']) for r in positions+influence}
    raw,verification=lengths.recover(source,args.cache_dir,selected,data)
    max_lag=manifest['settings']['max_lag']
    times={u:raw[u][:len(data.train_by_user[u])+1][-max_lag:] for u in selected}
    mapping={u:group_lags(t) for u,t in times.items()}
    history_ids={u:(list(data.train_by_user[u])+[data.valid_target[u]])[-max_lag:] for u in selected}
    user_groups=defaultdict(lambda:defaultdict(list))
    signal=defaultdict(lambda:defaultdict(float))
    for row in positions:
        u,lag=int(row['user']),int(row['lag'])
        assert int(row['item'])==history_ids[u][-lag]
        assert int(row['target'])==data.test_target[u]
        g=mapping[u][lag]
        user_groups[u][g].append((float(row['excess_cosine_z']),float(row['history_normalized_cluster_match'])))
        signal[u][g]+=max(float(row['excess_cosine_z']),0.0)
    output={}
    for metric,index in [('alignment',0),('preference',1)]:
        output[metric]={}
        for cohort in ['all_available','fixed_K16']:
            values=defaultdict(list)
            for u,groups in user_groups.items():
                if cohort=='fixed_K16' and len(times[u])<16:continue
                for g,items in groups.items():
                    values[g].append(float(np.mean([x[index] for x in items])))
            output[metric][cohort]=per_group_stats(values,args.seed+(0 if index==0 else 1000),args.bootstrap,args.device)
    effects=defaultdict(lambda:defaultdict(list))
    for row in influence:
        u,lag=int(row['user']),int(row['lag'])
        assert len(times[u])==int(row['history_length'])
        assert int(row['item'])==history_ids[u][-lag] and int(row['target'])==data.test_target[u]
        effects[u][mapping[u][lag]].append(float(row['margin_influence']))
    values=defaultdict(list)
    for groups in effects.values():
        for g,items in groups.items():values[g].append(float(np.mean(items)))
    output['influence']=per_group_stats(values,args.seed+2000,args.bootstrap,args.device)
    positive={u:s for u,s in signal.items() if sum(s.values())>0}
    original=read_rows(folder/'information_coverage_per_user.csv')
    assert set(positive)=={int(r['user']) for r in original},'Coverage cohort changed'
    for row in original:
        assert np.isclose(sum(positive[int(row['user'])].values()),float(row['positive_signal_total']))
    coverage=[]
    max_groups=max((max(mapping[u].values()) for u in positive),default=0)
    for k in range(1,max_groups+1):
        vals=[100*sum(v for g,v in s.items() if g<=k)/sum(s.values()) for s in positive.values()]
        a=torch.tensor(vals,dtype=torch.float64,device=args.device)
        q=torch.quantile(a,torch.tensor([.25,.5,.75],dtype=a.dtype,device=args.device))
        coverage.append({'group':k,'mean':float(a.mean()),'q25':float(q[0]),
                         'median':float(q[1]),'q75':float(q[2]),'n_users':len(vals)})
    output['coverage']=coverage
    report=json.loads(Path(source.get('base_report', root/'outputs/amazon_group_lengths'/(category+'.json'))).read_text())
    report['five_figure_statistics']=output
    report['five_figure_method']={
        'x':'recent-first 30-day consecutive-gap temporal group',
        'alignment':'mean excess_cosine_z within user/group, then equal mean across users',
        'preference':'mean original semantic-cluster excess match within user/group; original per-user 1/K chance retained',
        'influence':'mean original single-item deletion margin effect within user/group; NOT joint group ablation',
        'coverage':'sum positive original excess cosine z per group; cumulative recent groups / all positive signal in observed history',
        'confidence_interval':'95% user bootstrap for alignment/preference/influence; coverage shading is user IQR',
        'cohorts':'original test history and max100 items; fixed_K16 means at least16 original items, not16 groups',
        'main_min_users':args.min_users,'supplement':'same influence estimates, all n>=1',
        'gpu':torch.cuda.get_device_name(0),'physical_gpu':args.physical_gpu,'bootstrap_repetitions':args.bootstrap,
        'position_source':str(positions_path),'position_source_sha256':hashlib.sha256(positions_path.read_bytes()).hexdigest(),
        'influence_source_sha256':hashlib.sha256(Path(diagnostic['model_item_influence_source']).read_bytes()).hexdigest(),
        'timestamp_verification':verification,'training_repeated':False}
    (args.output/(category+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(category,'saved group statistics on',args.device,flush=True)
    return output


def plot(reports,args):
    from period_plot_style import LINESTYLES, MARKER
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':10,'pdf.fonttype':42,'svg.fonttype':'path'})
    styles={'All_Beauty':('All Beauty','#0072B2','o'),'Baby_Products':('Baby Products','#D55E00','s'),
            'Sports_and_Outdoors':('Sports and Outdoors','#009E73','^'),'Toys_and_Games':('Toys and Games','#CC79A7','D')}
    styles=getattr(args,'styles',styles)
    descriptions=[('figure1_alignment','alignment','Semantic alignment (excess cosine z)'),
                  ('figure2_preference','preference','Semantic-cluster excess match (percentage points)'),
                  ('figure3_influence','influence','Mean item-deletion margin within period'),
                  ('figure4_coverage','coverage','Cumulative positive signal (%)'),
                  ('figureS1_influence_all_groups','influence_all','Mean item-deletion margin within period')]
    all_rows=[]
    for name,metric,ylabel in descriptions:
        panels=metric in ['alignment','preference']
        fig,axes=plt.subplots(1,2 if panels else 1,figsize=(10.4,4.3) if panels else (8.1,4.5),squeeze=False)
        for ax,cohort in zip(axes[0],['all_available','fixed_K16'] if panels else ['all_available']):
            for category,stats in reports.items():
                kind='influence' if metric=='influence_all' else metric
                rows=stats[kind][cohort] if panels else stats[kind]
                for r in rows:all_rows.append({'figure':name,'subset':category,'cohort':cohort,**r})
                selected=[r for r in rows if r['n_users']>=(1 if metric in ['coverage','influence_all'] else args.min_users)]
                label,color,marker=styles[category]
                if not selected:continue
                scale=100 if metric=='preference' else 1
                x=[r['group'] for r in selected];y=[r['mean']*scale for r in selected]
                ax.plot(x,y,label=label,color=color,marker=marker,ls=LINESTYLES[category],markevery=max(1,len(x)//9),**MARKER)
                low='q25' if metric=='coverage' else 'ci_low';high='q75' if metric=='coverage' else 'ci_high'
                shaded=[r for r in selected if r.get(low) is not None]
                if shaded:ax.fill_between([r['group'] for r in shaded],[r[low]*scale for r in shaded],
                                         [r[high]*scale for r in shaded],color=color,alpha=.08 if metric=='coverage' else .11,linewidth=0,zorder=1)
            ax.set_xlabel('Time period (1 = most recent; gap > 30 days starts a new period)')
            ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.25)
            ax.set_title(('All available histories' if cohort=='all_available' else 'Original history ≥ 16 items') if panels else name.replace('_',' ').replace('groups','periods'))
            if metric=='coverage':ax.set_ylim(0,102);ax.axhline(80,color='gray',linestyle=':',linewidth=.8)
            else:ax.axhline(0,color='gray',linewidth=.6)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False)
        note='Test split · 100-item cap · period means within user before averaging · 95% user-bootstrap CI'
        if metric=='coverage':note='Test split · cumulative whole temporal periods · shading: user IQR · zero-positive-signal users excluded'
        if metric.startswith('influence'):note+='\nAverage single-item deletion effect; this is not a whole-period ablation.'
        if metric=='influence_all':note+=' Includes low-sample periods.'
        fig.text(.5,.01,note,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.08,1,.91))
        for ext in ['png','pdf','svg']:fig.savefig(args.output/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    fields=sorted({k for r in all_rows for k in r})
    with (args.output/'plotted_data.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=root/'outputs/grouped_preliminary')
    p.add_argument('--cache-dir',type=Path,default=root.parents[1]/'cache')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--min-users',type=int,default=100)
    p.add_argument('--seed',type=int,default=25252)
    p.add_argument('--sources',type=Path,default=root.parent/'preliminary/aggregated/sources.json',
                   help='sources.json from the matching preliminary run')
    p.add_argument('--physical-gpu',type=int,default=1)
    args=p.parse_args();args.device='cuda:0'
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Launch with CUDA_VISIBLE_DEVICES=0; exactly one visible GPU required')
    torch.cuda.set_device(0)
    import pyarrow as pa
    pa.set_cpu_count(4)
    sys.path.insert(0,str(root.parent/'ver4'))
    args.output.mkdir(parents=True,exist_ok=True)
    sources=json.loads(args.sources.read_text())['sources']
    reports={}
    for source,category in zip(sources,['All_Beauty','Baby_Products','Sports_and_Outdoors','Toys_and_Games']):
        reports[category]=calculate(dict(source,category=category),args,root)
    plot(reports,args)
    (args.output/'run_status.json').write_text(json.dumps({'status':'complete','physical_gpu':args.physical_gpu,
        'figures':5,'formats':['png','pdf','svg'],'subsets':list(reports),'bootstrap':args.bootstrap},indent=2))
    (args.output/'README.md').write_text('# Five preliminary figures by temporal group\n\n'
        'Figures 1–4 plus the all-group influence supplement are provided as PNG/PDF/SVG. Each subset JSON contains group-length statistics and all five figure estimates. plotted_data.csv includes all computed groups, including those hidden by the main n≥100 filter.\n\n'
        'Groups split only after consecutive gaps >30 days. Figure2 retains original semantic cluster labels and chance baseline. Figure3/S1 average individual item-deletion effects inside each time group; they do not measure deleting the entire group. Fixed-K16 means ≥16 original items. Raw data and original figures are unchanged.\n')
    print('COMPLETE: five figures, four subset JSONs, plotted data CSV',flush=True)


if __name__=='__main__':main()
