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


def finalize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets=['Steam','MovieLens1M','Yelp19','Yelp23']
    reports={}
    sys.path.insert(0,str(ROOT.parent/'ver4'))
    grouping=load_module('grouping_map',ROOT/'plot_grouped_preliminary.py')
    for name in datasets:
        report=json.loads((root/name/(name+'.json')).read_text())
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
    save(root/'completion.json',{'status':'complete','datasets':datasets,'aggregate_figures':str(destination)})
    print('All four datasets complete; aggregate figures saved:',destination,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    while True:
        status={n:json.loads((args.root/n/'status.json').read_text()).get('status')
                if (args.root/n/'status.json').exists() else 'queued' for n in STYLES}
        if all(x=='complete' for x in status.values()):finalize(args.root);return
        if 'failed' in status.values():
            save(args.root/'completion.json',{'status':'waiting_for_failed_job_repair','datasets':status});print(status,flush=True);return
        if not args.watch:print(status);return
        time.sleep(20)


if __name__=='__main__':main()
