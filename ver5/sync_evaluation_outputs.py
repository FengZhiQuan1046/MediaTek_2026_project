"""Mirror only small scores/configuration/logs; backfill every logged evaluation."""
import argparse
import json
from pathlib import Path
import re
import shutil
import time

BLOCK = re.compile(
    r'(?P<time>\d{4}-\d\d-\d\d [\d:,]+) \| INFO \| \n'
    r'=+ (?P<label>[^\n]+?) =+\n'
    r'stage=(?P<stage>\S+) \| epoch=(?P<epoch>\d+) \| step=(?P<step>\d+) \| users=(?P<users>\d+)/(?P<total>\d+)\n'
    r'NDCG\s*\| @5 (?P<n5>[\d.eE+-]+) \| @10 (?P<n10>[\d.eE+-]+)\n'
    r'Recall\s*\| @5 (?P<r5>[\d.eE+-]+) \| @10 (?P<r10>[\d.eE+-]+)\n'
    r'Hit\s*\| @5 (?P<h5>[\d.eE+-]+) \| @10 (?P<h10>[\d.eE+-]+)\n=+'
)


def evaluations(text, dataset):
    rows=[]
    for match in BLOCK.finditer(text):
        v=match.groupdict()
        rows.append(dict(dataset=dataset,timestamp=v['time'],evaluation=v['label'],
                         stage=v['stage'],epoch=int(v['epoch']),step=int(v['step']),
                         evaluated_users=int(v['users']),total_users=int(v['total']),
                         metrics={name:float(v[key]) for name,key in [
                             ('ndcg@5','n5'),('ndcg@10','n10'),('recall@5','r5'),
                             ('recall@10','r10'),('hit@5','h5'),('hit@10','h10')]}))
    return rows


def atomic_write(path, text):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(text,encoding='utf-8')
    temp.replace(path)


def sync(source, target):
    counts={}
    for dataset in ['steam','yelp23','yelp19']:
        folder=target/dataset/source.name
        folder.mkdir(parents=True,exist_ok=True)
        manifest=source/'launch_manifest.json'
        if manifest.exists():shutil.copy2(manifest,folder/manifest.name)
        logs=list((source/dataset).glob('train_*.log'))
        for log in logs:
            text=log.read_text(encoding='utf-8',errors='replace')
            atomic_write(folder/log.name,text)
            rows=evaluations(text,dataset)
            counts[dataset]=len(rows)
            atomic_write(folder/'evaluations.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            configs=re.findall(r'EXPERIMENT_CONFIG (\{[^\n]+\})',text)
            if configs:
                atomic_write(folder/'config.json',json.dumps(json.loads(configs[-1]),indent=2)+'\n')
            if rows:atomic_write(folder/'latest_evaluation.json',json.dumps(rows[-1],indent=2)+'\n')
        # Allowlist: never copy weights, raw data, vectors, or console progress.
        for name in ['metrics.json',f'{dataset}_scores.json','status.txt']:
            p=source/dataset/name
            if p.exists():
                try:
                    content=p.read_text()
                    if name.endswith('.json'):json.loads(content)
                    atomic_write(folder/name,content)
                except json.JSONDecodeError:
                    pass  # Trainer may currently be writing final metrics.
        atomic_write(folder/'sync_status.json',json.dumps({
            'source':str(source/dataset),'target':str(folder),
            'evaluations_saved':counts.get(dataset,0),'updated_unix':time.time(),
            'training_finished':(source/dataset/'status.txt').exists(),
            'metric_precision':'six decimal places as recorded by existing trainer; final scores retain original precision'
        },indent=2)+'\n')
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('target',type=Path)
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    while True:
        counts=sync(a.source,a.target)
        if not a.watch:
            print(json.dumps(counts));break
        if all((a.source/d/'status.txt').exists() for d in ['steam','yelp23','yelp19']):
            print('All runs finished and synchronized',counts,flush=True);break
        time.sleep(5)


if __name__=='__main__':main()
