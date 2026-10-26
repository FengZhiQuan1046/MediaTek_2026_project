"""Download pinned MAPLE Yelp releases into workspace cache and normalize once."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, default=ROOT.parent/'cache')
    p.add_argument('--datasets', nargs='+', default=['yelp19','yelp23'], choices=['yelp19','yelp23'])
    p.add_argument('--repo-id', default='NanaEilish/maple_nete_format_data')
    args = p.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with (args.cache_dir/'prepare_hf_yelp.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        revision = HfApi().dataset_info(args.repo_id).sha
        for dataset in args.datasets:
            output = args.cache_dir/'normalized'/f'{dataset}_unique.jsonl'
            if output.exists() and output.with_suffix('.source.json').exists():
                print(f'Ready: {output}', flush=True); continue
            raw = args.cache_dir/'hf_yelp'/dataset
            archive = Path(hf_hub_download(args.repo_id, 'yelp.tar.gz' if dataset=='yelp19' else 'yelp23.tar.gz',
                                         repo_type='dataset', revision=revision, local_dir=raw))
            extracted = raw/'extracted'; extracted.mkdir(exist_ok=True)
            with tarfile.open(archive) as tar:
                tar.extractall(extracted, filter='data')
            sources = list(extracted.rglob('reviews.pickle'))
            if len(sources)!=1: raise RuntimeError(f'Ambiguous reviews.pickle: {sources}')
            normalized = raw/'interactions.jsonl'
            subprocess.run([sys.executable, str(ROOT/'ver5/prepare_yelp19.py'), str(sources[0]),str(normalized)],check=True)
            output.parent.mkdir(exist_ok=True)
            subprocess.run([sys.executable,str(ROOT/'ver5/deduplicate_yelp_interactions.py'),str(normalized),str(output)],check=True)
            output.with_suffix('.source.json').write_text(json.dumps(dict(repo_id=args.repo_id,revision=revision,
                archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))
if __name__=='__main__': main()
