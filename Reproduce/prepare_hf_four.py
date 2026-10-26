"""Download full HF releases into workspace cache; normalize with ver5 code."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from huggingface_hub import HfApi, hf_hub_download

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,default=ROOT.parent/'cache')
    p.add_argument('--datasets',nargs='+',default=['yelp19','yelp23','movielens-1m','steam'])
    a=p.parse_args();a.cache_dir.mkdir(parents=True,exist_ok=True)
    for dataset in a.datasets:
        if dataset in ['yelp19','yelp23']:
            subprocess.run([sys.executable,str(ROOT/'Reproduce/prepare_hf_yelp.py'),'--cache-dir',str(a.cache_dir),'--datasets',dataset],check=True)
            continue
        if dataset=='movielens-1m':
            repo='nasserCha/movielens_rating_1m';files=['movies.dat','ratings.dat']
        elif dataset=='steam':
            repo='recommender-system/steam-review-and-bundle-dataset';files=['steam_games.json.gz','steam_reviews.json.gz']
        else:raise ValueError(dataset)
        raw=a.cache_dir/'hf_datasets'/dataset;raw.mkdir(parents=True,exist_ok=True)
        with (raw/'prepare.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if (raw/'ready.json').exists():continue
            revision=HfApi().dataset_info(repo).sha
            paths=[Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision,local_dir=raw)) for name in files]
            if dataset=='steam':
                subprocess.run([sys.executable,str(ROOT/'ver5/prepare_hf_steam.py'),str(raw),str(a.cache_dir/'normalized/steam.csv')],check=True)
            else:
                ratings=raw/'ratings.dat'
                with ratings.open() as f:count=sum(1 for _ in f)
                if count!=1000209:raise RuntimeError(f'Unexpected ML1M rating count: {count}')
                # Verify the HF mirror against the existing original release, when present.
                original=a.cache_dir/'raw/movielens-1m/ml-1m'
                for path in paths:
                    if (original/path.name).exists() and sha(path)!=sha(original/path.name):
                        raise RuntimeError(f'HF mirror differs from original: {path.name}')
            (raw/'ready.json').write_text(json.dumps(dict(repo_id=repo,revision=revision,sha256={p.name:sha(p) for p in paths}),indent=2))
            print('Ready',dataset,flush=True)
if __name__=='__main__':main()
