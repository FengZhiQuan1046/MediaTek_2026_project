"""Train the original single-long-branch preliminary model on a prepared cache."""
import json
from pathlib import Path
import pickle
import sys

root=Path(__file__).resolve().parents[1]
job=json.loads(Path(sys.argv[1]).read_text())
sys.argv=[sys.argv[0],*sys.argv[2:]]
sys.path.insert(0,str(root/'ver4'))
from src import train_mamba_rl as trainer


def load_cached(args,logger):
    path=Path(job['cache_path'])
    with path.open('rb') as stream:data=pickle.load(stream)
    logger.info('PREPARED_DATED_CACHE %s',path)
    return data,path

trainer.load_recommendation_data_cached=load_cached
trainer.main()
