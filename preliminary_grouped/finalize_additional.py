"""Wait for the two dataset queues, then render all four extra curves together."""
import argparse
import gzip
import pickle
import sys
import json
from pathlib import Path
import time
from types import SimpleNamespace

from additional_datasets import ROOT, STYLES, load_module, compact, save


def finalize(root, mamba_gate=None, mamba_decay=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets=['Steam','MovieLens1M','Yelp19','Yelp23']
    reports={}
    observed_gates=set()
    observed_decays=set()
    sys.path.insert(0,str(ROOT.parent/'ver4'))
    grouping=load_module('grouping_map',ROOT/'plot_grouped_preliminary.py')
    for name in datasets:
        report=json.loads((root/name/(name+'.json')).read_text())
        observed_gates.add(report.get('mamba_gate'))
        observed_decays.add(report.get('mamba_decay'))
        if mamba_gate is not None and report.get('mamba_gate') != mamba_gate:
            raise RuntimeError(
                f"{name} gate mismatch: expected {mamba_gate}, got {report.get('mamba_gate')}"
            )
        if mamba_decay is not None and report.get('mamba_decay') != mamba_decay:
            raise RuntimeError(
                f"{name} decay mismatch: expected {mamba_decay}, got {report.get('mamba_decay')}"
            )
        group_map=root/name/'intermediate/time_group_map.json.gz'
        if not group_map.exists():
            job=json.loads((root/name/'intermediate/job.json').read_text())
            with open(job['cache_path'],'rb') as f:data=pickle.load(f)
            with gzip.open(root/name/'intermediate/timestamps.json.gz','rt') as f:times=json.load(f)
            maps={u:grouping.group_lags(ts[:len(data.train_by_user[int(u)])+1][-100:]) for u,ts in times.items()}
            with gzip.open(group_map,'wt') as f:json.dump(maps,f)
        report['reusable_intermediates']['time_group_map']=str(group_map)
        save(root/name/(name+'.json'),report)
        reports[name]=report['five_figure_statistics']
    if len(observed_gates) != 1 or None in observed_gates:
        raise RuntimeError(f'Cannot aggregate mixed or missing Mamba gates: {observed_gates}')
    observed_gate=next(iter(observed_gates))
    if mamba_gate is None:mamba_gate=observed_gate
    if len(observed_decays)!=1:
        raise RuntimeError(f'Cannot aggregate mixed Mamba decays: {observed_decays}')
    if mamba_decay is None:mamba_decay=next(iter(observed_decays))
    destination=root/'aggregated';destination.mkdir(exist_ok=True)
    grouped=load_module('aggregate_grouped',ROOT/'plot_grouped_preliminary.py')
    grouped.plot(reports,SimpleNamespace(output=destination,min_users=100,styles=STYLES))
    compact(reports,destination)
    means=[]
    for name in datasets:
        count=json.loads((root/name/'group_count_summary.json').read_text())
        means.append(count['sampled_preliminary_cohort_capped']['mean_groups_per_user'])
    fig,ax=plt.subplots(figsize=(6.5,3.8))
    ax.bar(datasets,means,color=[STYLES[n][1] for n in datasets])
    for i,value in enumerate(means):ax.text(i,value+.1,f'{value:.3f}',ha='center')
    ax.set_ylabel('Mean temporal periods per user')
    ax.set_title('Observed test history: up to100 items; new period after gap >30 days')
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(destination/('mean_groups_per_user.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    save(root/'completion.json',{'status':'complete','datasets':datasets,'mamba_gate':mamba_gate,
                                 'mamba_decay':mamba_decay,
                                 'aggregate_figures':str(destination),
                                 'combined_figure_stem':'figure0_abc_row'})
    print('All four datasets complete; aggregate figures saved:',destination,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--watch',action='store_true')
    p.add_argument('--mamba-gate',type=float)
    p.add_argument('--mamba-decay',type=float)
    args=p.parse_args()
    while True:
        status={n:json.loads((args.root/n/'status.json').read_text()).get('status')
                if (args.root/n/'status.json').exists() else 'queued' for n in STYLES}
        if all(x=='complete' for x in status.values()):
            finalize(args.root,args.mamba_gate,args.mamba_decay);return
        if 'failed' in status.values():
            save(args.root/'completion.json',{'status':'waiting_for_failed_job_repair','datasets':status});print(status,flush=True);return
        if not args.watch:print(status);return
        time.sleep(20)


if __name__=='__main__':main()
