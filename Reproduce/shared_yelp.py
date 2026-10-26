"""Exact ver5 dataset filtering/splits and SIGMA evaluation output schema."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ver5'))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

from src.data import build_data, MIN_INTERACTIONS
from src.data_mamba_rl import load_recommendation_data
from src.train_yelp_steam import load_steam_events
EvaluationWriter=module('evaluation_outputs',ROOT/'ver5/src/evaluation_outputs.py').EvaluationWriter

def load_shared(args,logger):
    identity=dict(dataset=args.dataset,data_path=str(Path(args.data_path).resolve()) if args.data_path else None,
                  max_events=args.max_events,min_rating=args.min_rating,min_interactions=MIN_INTERACTIONS,schema_version=2)
    digest=hashlib.sha1(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    artifact=Path(args.cache_dir)/'mamba_multi_agent_data'/f"{args.dataset.replace(':','_').replace('/','_')}_{digest}.pkl"
    artifact.parent.mkdir(parents=True,exist_ok=True)
    with artifact.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if artifact.exists() and not args.refresh_data_cache:
            with artifact.open('rb') as f:data=pickle.load(f)
        else:
            if args.dataset=='steam':data=build_data(load_steam_events(args.data_path,args.max_events,args.min_rating))
            else:data=load_recommendation_data(args.dataset,args.data_path,args.cache_dir,args.max_events,args.min_rating)
            tmp=artifact.with_suffix('.tmp')
            with tmp.open('wb') as f:pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(artifact)
    h=hashlib.sha256()
    with artifact.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    info=dict(artifact=str(artifact),cache_sha256=h.hexdigest(),bytes=artifact.stat().st_size,
        num_users=data.num_users,num_items=data.num_items,validation_users=len(data.valid_target),test_users=len(data.test_target),
        train_interactions=sum(map(len,data.train_by_user.values())),data_protocol='ver5 loaders and build_data without changes',
        filtering=identity,loader_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'ver5/src/data.py',ROOT/'ver5/src/data_mamba_rl.py',ROOT/'ver5/src/train_yelp_steam.py']})
    (Path(args.output_dir)/'data_info.json').write_text(json.dumps(info,indent=2))
    logger.info('VER5_SHARED_DATA %s',json.dumps(info))
    return data,artifact
