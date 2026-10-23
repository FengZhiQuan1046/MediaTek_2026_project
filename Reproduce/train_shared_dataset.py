"""Run a reproduction on the exact cached split used by ver5; log every eval."""
import argparse
import importlib.util
import inspect
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--reproduction',choices=['Mamba4Rec','ReSID','SASRec','BERT4Rec','SIGMA'],required=True)
    options, remaining=parser.parse_known_args()
    model_root=ROOT/'Reproduce'/options.reproduction
    sys.path.insert(0,str(model_root))
    import train as trainer
    spec=importlib.util.spec_from_file_location('evaluation_outputs',ROOT/'ver5/src/evaluation_outputs.py')
    outputs=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(outputs)
    original_parse=trainer.parse_args
    original_eval=trainer.evaluate
    writer=None

    def parse_args():
        nonlocal writer
        args=original_parse()
        output=Path(args.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        config=vars(args).copy()
        config.update({'reproduction':options.reproduction,'save_model_weights':False,
                       'data_protocol':'Exact existing ver5 InteractionData pickle, no re-split',
                       'yelp_ordering':'source-row order; release has no timestamps'})
        (output/'config.json').write_text(json.dumps(config,indent=2))
        writer=outputs.EvaluationWriter(output,args.dataset)
        return args

    def load_shared(args,logger):
        identity={'dataset':args.dataset,
                  'data_path':str(Path(args.data_path).resolve()) if args.data_path else None,
                  'max_events':args.max_events,'min_rating':args.min_rating,
                  'min_interactions':5,'schema_version':2}
        digest=hashlib.sha1(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
        name=args.dataset.replace(':','_').replace('/','_')
        artifact=Path(args.cache_dir)/'mamba_multi_agent_data'/f'{name}_{digest}.pkl' 
        if not artifact.is_file():
            raise FileNotFoundError(f'Exact ver5 cache is required: {artifact}')
        if args.refresh_data_cache:
            raise ValueError('Shared-split runs cannot refresh or replace the ver5 cache')
        with artifact.open('rb') as stream:data=pickle.load(stream)
        checksum=hashlib.sha256()
        with artifact.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                checksum.update(chunk)
        info={'artifact':str(artifact),'bytes':artifact.stat().st_size,
              'cache_sha256':checksum.hexdigest(),
              'num_users':data.num_users,'num_items':data.num_items,
              'validation_users':len(data.valid_target),'test_users':len(data.test_target),
              'train_interactions':sum(map(len,data.train_by_user.values()))}
        (Path(args.output_dir)/'data_info.json').write_text(json.dumps(info,indent=2))
        logger.info('EXACT_VER5_SHARED_CACHE %s',json.dumps(info,sort_keys=True))
        return data,artifact

    def evaluate(model,data,split,*eval_args,**eval_kwargs):
        context=inspect.currentframe().f_back.f_locals
        epoch=context.get('epoch') if split=='valid' else context.get('best_epoch')
        batches=context.get('batches',context.get('batches_per_epoch'))
        step=epoch*batches if epoch is not None and batches is not None else None
        metrics=original_eval(model,data,split,*eval_args,**eval_kwargs)
        metrics['total_users']=len(data.valid_target if split=='valid' else data.test_target)
        writer.write('VALIDATION' if split=='valid' else 'FINAL TEST',metrics,
                     'recommender' if split=='valid' else 'best_validation_model',epoch,step)
        return metrics

    trainer.parse_args=parse_args
    trainer.load_data_cached=load_shared
    trainer.evaluate=evaluate
    sys.argv=[sys.argv[0],*remaining]
    trainer.main()
    # Preserve augmented config after the trainer's final standard config write.
    path=writer.output/'config.json'
    config=json.loads(path.read_text())
    config.update({'reproduction':options.reproduction,'save_model_weights':False,
                   'data_protocol':'Exact existing ver5 InteractionData pickle, no re-split',
                   'yelp_ordering':'source-row order; release has no timestamps'})
    path.write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
