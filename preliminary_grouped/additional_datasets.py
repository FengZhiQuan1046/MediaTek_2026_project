"""Reusable dated preliminary workflow: prepare -> train once -> signals -> plots."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
CACHE=REPO.parent/'cache'
PYTHON=REPO.parent/'miniconda3/envs/py31014/bin/python'
STYLES={'Steam':('Steam','#6A3D9A','v'),'MovieLens1M':('MovieLens 1M','#8C564B','P'),
        'Yelp19':('Yelp19','#E69F00','X'),
        'Yelp23':('Yelp23','#4D4D4D','h')}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sql_string(path):return "'"+str(path).replace("'","''")+"'"


def prepare(name,output):
    """Large normalized events and SQL workspaces stay in cache."""
    import duckdb
    work=CACHE/'grouped_preliminary'/name;work.mkdir(parents=True,exist_ok=True)
    job_file=output/'intermediate/job.json'
    if job_file.exists():return json.loads(job_file.read_text())
    db=duckdb.connect(str(work/'preparation.duckdb'))
    db.execute("SET threads=4")
    db.execute("SET memory_limit='12GB'")
    db.execute(f"SET temp_directory={sql_string(work/'tmp')}")
    provenance={'label':name,'original_datasets_unchanged':True}
    if name=='Steam':
        source=CACHE/'normalized/steam.csv'
        db.execute(f"CREATE OR REPLACE TABLE events AS SELECT user_id,item_id,epoch_ms(strptime(date,'%Y-%m-%d'))::BIGINT AS time_ms,item_text,row_number() OVER() AS source_order FROM read_csv({sql_string(source)},header=true,all_varchar=true)")
        dataset='steam'
        cache_path=CACHE/'mamba_multi_agent_data/steam_e4cf210a33ca.pkl'
        provenance.update(source=str(source),cache_reused=True)
    elif name=='MovieLens1M':
        from huggingface_hub import hf_hub_download
        raw=CACHE/'raw/dated_analysis/movielens-1m'
        for filename in ['movies.dat','ratings.dat']:
            if not (raw/filename).exists():hf_hub_download('nasserCha/movielens_rating_1m',filename,repo_type='dataset',local_dir=str(raw),cache_dir=str(CACHE/'hub'))
        movies={}
        with (raw/'movies.dat').open(encoding='latin-1') as stream:
            for line in stream:
                movie,title,genres=line.rstrip('\n').split('::',2);movies[movie]=title+' Genres: '+genres
        normalized=work/'events.csv'
        with normalized.open('w',newline='') as target,(raw/'ratings.dat').open() as source:
            writer=csv.writer(target);writer.writerow(['user_id','item_id','time_ms','item_text','source_order'])
            for index,line in enumerate(source):
                user,item,rating,ts=line.strip().split('::')
                if float(rating)>=4:writer.writerow([user,item,int(ts)*1000,movies.get(item,item),index])
        db.execute(f"CREATE OR REPLACE TABLE events AS SELECT user_id::VARCHAR user_id,item_id::VARCHAR item_id,time_ms::BIGINT time_ms,item_text,source_order::BIGINT source_order FROM read_csv({sql_string(normalized)},header=true,all_varchar=true)")
        dataset='movielens-1m'
        cache_path=work/'interaction_data.pkl'
        provenance.update(source=str(raw),rating_minimum=4,source_repo='nasserCha/movielens_rating_1m')
    else:
        source=CACHE/'normalized'/('yelp19_unique.jsonl' if name=='Yelp19' else 'yelp23_unique.jsonl')
        historical=name=='Yelp19'
        raw=(CACHE/'raw/dated_analysis/yelp_historical/dataset_challenge/yelp_review.csv') if historical else (CACHE/'raw/dated_analysis/yelp_recent/review.parquet')
        if not raw.exists():raise FileNotFoundError(f'Dated HF download is not complete: {raw}')
        raw_reader=(f"read_csv({sql_string(raw)},header=true,all_varchar=true,max_line_size=16777216)") if historical else f"read_parquet({sql_string(raw)})"
        db.execute(f"CREATE OR REPLACE TABLE requested AS SELECT user_id::VARCHAR user_id,item_id::VARCHAR item_id,rating::DOUBLE rating,item_text,timestamp::BIGINT source_order FROM read_json_auto({sql_string(source)},maximum_object_size=16777216) WHERE rating>=4")
        # Keep only true dates that can be uniquely linked to the original IDs
        # and rating. Ambiguous/missing joins are counted, never guessed.
        db.execute(f"CREATE OR REPLACE TABLE raw_dates AS SELECT user_id::VARCHAR user_id,business_id::VARCHAR item_id,stars::DOUBLE rating,epoch_ms(try_cast(date AS TIMESTAMP))::BIGINT time_ms FROM {raw_reader} WHERE stars::DOUBLE>=4")
        db.execute("CREATE OR REPLACE TABLE date_keys AS SELECT user_id,item_id,rating,min(time_ms) time_ms,count(DISTINCT time_ms) date_count FROM raw_dates WHERE time_ms IS NOT NULL GROUP BY user_id,item_id,rating")
        total=db.execute('SELECT count(*) FROM requested').fetchone()[0]
        unique=db.execute('SELECT count(*) FROM requested r JOIN date_keys d USING(user_id,item_id,rating) WHERE date_count=1').fetchone()[0]
        ambiguous=db.execute('SELECT count(*) FROM requested r JOIN date_keys d USING(user_id,item_id,rating) WHERE date_count>1').fetchone()[0]
        assert unique>0,(name,'No matching dated records')
        db.execute('CREATE OR REPLACE TABLE events AS SELECT r.user_id,r.item_id,d.time_ms,r.item_text,r.source_order FROM requested r JOIN date_keys d USING(user_id,item_id,rating) WHERE d.date_count=1 ORDER BY r.source_order')
        dataset='yelp19-dated-matched' if historical else 'yelp23-dated-matched'
        cache_path=work/'interaction_data.pkl'
        provenance.update(source=str(source),dated_mirror=str(raw),requested_positive_rows=total,
            uniquely_dated_rows=unique,ambiguous_rows_excluded=ambiguous,missing_rows_excluded=total-unique-ambiguous,
            uniquely_dated_fraction=unique/total,scope='Original MAPLE positive user-item-rating rows with an unambiguous true date in the HF mirror; NOT the full original MAPLE population',
            dated_mirror_release='Yelp2018 historical mirror' if historical else 'Yelp Open Dataset recent mirror',
            timestamp_order='chronological; old source-row-index split is not reused')
    # Match the same simultaneous one-pass threshold, not iterative k-core.
    db.execute('CREATE OR REPLACE TABLE filtered AS WITH u AS (SELECT user_id FROM events GROUP BY user_id HAVING count(*)>=5), i AS (SELECT item_id FROM events GROUP BY item_id HAVING count(*)>=5) SELECT e.* FROM events e JOIN u USING(user_id) JOIN i USING(item_id) ORDER BY e.source_order')
    filtered=work/'filtered_timestamps.parquet'
    db.execute(f"COPY filtered TO {sql_string(filtered)} (FORMAT PARQUET,COMPRESSION ZSTD)")
    sys.path.insert(0,str(REPO/'ver4'))
    from src.data import build_data
    if not cache_path.exists():
        rows=db.execute('SELECT user_id,item_id,time_ms,item_text FROM events ORDER BY source_order').fetchall()
        data=build_data(rows)
        with cache_path.open('wb') as f:pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    else:
        with cache_path.open('rb') as f:data=pickle.load(f)
    provenance.update(filtered_interactions=db.execute('SELECT count(*) FROM filtered').fetchone()[0],
        filtered_users=db.execute('SELECT count(DISTINCT user_id) FROM filtered').fetchone()[0],
        min_timestamp_ms=db.execute('SELECT min(time_ms) FROM filtered').fetchone()[0],
        max_timestamp_ms=db.execute('SELECT max(time_ms) FROM filtered').fetchone()[0])
    assert provenance['filtered_users']==data.num_users
    job={'name':name,'dataset':dataset,'cache_path':str(cache_path),'filtered_timestamps':str(filtered),
         'cache_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest(),'provenance':provenance}
    save(job_file,job);save(output/'provenance.json',provenance)
    print('PREPARED',name,provenance,flush=True)
    db.close();return job


def recover(job,selected,data):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    table=pq.read_table(job['filtered_timestamps'],columns=['user_id','item_id','time_ms','source_order'])
    all_users=sorted(pc.unique(table['user_id']).to_pylist())
    assert len(all_users)==data.num_users
    numeric={all_users[u]:u for u in selected}
    table=table.filter(pc.is_in(table['user_id'],value_set=pa.array(list(numeric))))
    values=table.to_pydict();events=defaultdict(list)
    for user,item,ts,order in zip(values['user_id'],values['item_id'],values['time_ms'],values['source_order']):
        events[numeric[user]].append((ts,order,item))
    histories={};item_mapping={};reverse_mapping={}
    for user in sorted(selected):
        ordered=sorted(events[user],key=lambda x:(x[0],x[1]))
        ids=list(data.train_by_user[user])+([data.valid_target[user]] if user in data.valid_target else [])+([data.test_target[user]] if user in data.test_target else [])
        assert len(ordered)==len(ids),(job['name'],user,'history count mismatch')
        for item,(_,_,raw) in zip(ids,ordered):
            assert item_mapping.setdefault(item,raw)==raw,(job['name'],user,'item order mismatch')
            assert reverse_mapping.setdefault(raw,item)==item
        histories[user]=[r[0] for r in ordered]
    return histories,{'source':job['filtered_timestamps'],'verified_users':len(histories),'verified_items':len(item_mapping),'true_timestamps':True}


def train(job,job_file,output,epochs,mamba_gate,mamba_decay=None):
    influence=output/'intermediate/training'/job['dataset'].replace(':','_')/'history_item_influence.csv'
    complete=output/'intermediate/training/training_complete.json'
    if influence.exists() and complete.exists():
        previous=json.loads(complete.read_text())
        if previous.get('mamba_gate')!=mamba_gate or previous.get('mamba_decay')!=mamba_decay:
            raise RuntimeError(f'Existing training uses different gate/decay: {complete}')
        print('REUSE trained-model influence',influence,flush=True);return
    prep=load_module('prepare_original',REPO/'preliminary/prepare_missing.py')
    prep.SUBSET_SETTINGS[job['dataset']]={'max_transitions':500000,'candidates':64,'popularity_alpha':0.0,'transition_beta':0.0}
    prep.ver4_cache_path=lambda *a:Path(job['cache_path'])
    original_run=subprocess.run
    def launch(command,*args,**kwargs):
        if command[0]=='bash' and command[1].endswith('run_mamba_rl.sh'):
            command=[str(PYTHON),str(ROOT/'train_additional.py'),str(job_file),
                '--dataset',job['dataset'],'--cache-dir',str(CACHE),'--device','cuda:0','--graph-device','cuda:0',*command[4:]]
        return original_run(command,*args,**kwargs)
    prep.subprocess.run=launch
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    old=sys.argv
    sys.argv=['prepare','--cache-dir',str(CACHE),'--output-dir',str(output/'intermediate/training'),
        '--datasets',job['dataset'],'--gpu-ids',gpu,'--python-bin',str(PYTHON),
        '--mamba-model-id','state-spaces/mamba-130m-hf','--lora-rank','16','--epochs',str(epochs),
        '--max-history','100','--influence-user-limit','2000','--validation-user-limit','0',
        '--seed','25252','--mamba-gate',str(mamba_gate),
        *(['--mamba-decay',str(mamba_decay)] if mamba_decay is not None else [])]
    try:prep.main()
    finally:subprocess.run=original_run;sys.argv=old
    save(output/'intermediate/training/training_complete.json',{
        'status':'complete','epochs':epochs,'mamba_gate':mamba_gate,
        'mamba_decay':mamba_decay,'model_weights_saved':False})


def compact(reports,destination):
    from period_plot_style import LINESTYLES, MARKER
    import matplotlib.pyplot as plt
    style=load_module('aggregate_style',REPO/'preliminary/aggregated/plot_aggregated.py')
    style.style(None)
    fig,axes=plt.subplots(1,3,figsize=(7.15,2.25))
    for ax,(metric,title,ylabel) in zip(axes,[('alignment','(a) Semantic alignment','Excess cosine (z)'),('influence','(b) Item influence','Margin change'),('coverage','(c) Signal coverage','Signal covered (%)')]):
        style.decorate(ax,title,ylabel,'Time period',metric!='coverage')
        for name,stats in reports.items():
            rows=stats[metric]['all_available'] if metric=='alignment' else stats[metric]
            rows=[r for r in rows if metric=='coverage' or r['n_users']>=100]
            if not rows:continue
            label,color,marker=STYLES[name]
            ax.plot([r['group'] for r in rows],[r['mean'] for r in rows],label=label,color=color,marker=marker,ls=LINESTYLES[name],markevery=max(1,len(rows)//9),**MARKER)
            lo,hi=('q25','q75') if metric=='coverage' else ('ci_low','ci_high')
            shade=[r for r in rows if r.get(lo) is not None]
            ax.fill_between([r['group'] for r in shade],[r[lo] for r in shade],[r[hi] for r in shade],color=color,alpha=.08 if metric=='coverage' else .11,linewidth=0,zorder=1)
        if metric=='coverage':
            ax.set_xlim(.5,20.5);ax.set_ylim(0,102);ax.set_xticks([1,5,10,15,20])
            ax.axhline(80,color='#777777',lw=.65,ls='--',zorder=1)
        ax.title.set_fontsize(8.5);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.tick_params(labelsize=7)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.998),
               ncol=min(4,len(reports)),frameon=False,columnspacing=1.6,handlelength=2.5)
    fig.subplots_adjust(left=.085,right=.985,top=.78,bottom=.21,wspace=.20)
    style.save(fig,destination,'figure0_abc_row',['png','pdf','svg'],400);plt.close(fig)


def analyze(job,output,args):
    import torch
    exp=load_module('original_experiment',REPO/'preliminary/experiment.py')
    grouped=load_module('grouped_figures',ROOT/'plot_grouped_preliminary.py')
    import amazon_group_lengths as gl
    settings=SimpleNamespace(cache_dir=str(CACHE),training_output_dir=str(output/'intermediate/training'),
        max_users=2000,max_lag=100,short_window=10,random_matches=20,reference_items=20000,
        normalization_pairs=100000,cluster_items=10000,clusters=32,bootstrap=2000,
        seed=25252,reverse_history_input=False,mamba_gate=args.mamba_gate,
        mamba_decay=args.mamba_decay)
    exp.ver4_cache_path=lambda *a:Path(job['cache_path'])
    exp.bootstrap_mean=lambda values,repetitions,seed: bootstrap_tuple(grouped,values,repetitions,seed)
    intermediate=output/'intermediate/signals';intermediate.mkdir(parents=True,exist_ok=True)
    signal_dir=intermediate/job['dataset'].replace(':','_')
    if not (intermediate/'manifest.json').exists():
        a,b,c,d,e,diagnostic=exp.analyse_dataset(job['dataset'],settings,intermediate)
        for name in ['alignment_summary.csv','information_coverage.csv','information_coverage_per_user.csv']:
            shutil.copy2(signal_dir/name,intermediate/name)
        diagnostic['model_item_influence_source']=str(output/'intermediate/training'/job['dataset'].replace(':','_')/'history_item_influence.csv')
        save(intermediate/'manifest.json',{'status':'complete','settings':vars(settings),'datasets':[diagnostic]})
    else:diagnostic=json.loads((intermediate/'manifest.json').read_text())['datasets'][0]
    selected=set(gl.csv_users(signal_dir/'alignment_per_position.csv'))|gl.csv_users(Path(diagnostic['model_item_influence_source']))
    with Path(job['cache_path']).open('rb') as f:data=pickle.load(f)
    timestamps,verification=recover(job,selected,data)
    # Reuse artifacts allow future thresholds/statistics without training or raw scans.
    timestamp_file=output/'intermediate/timestamps.json.gz'
    with gzip.open(timestamp_file,'wt') as stream:json.dump({str(u):ts for u,ts in timestamps.items()},stream)
    full={u:ts[:len(data.train_by_user[u])+1] for u,ts in timestamps.items()}
    truncated={u:ts[-100:] for u,ts in full.items()}
    group_map=output/'intermediate/time_group_map.json.gz'
    with gzip.open(group_map,'wt') as stream:
        json.dump({str(u):grouped.group_lags(ts) for u,ts in truncated.items()},stream)
    users=gl.csv_users(signal_dir/'alignment_per_position.csv')
    report={'dataset':job['dataset'],'subset':job['name'],'split':'test','grouping':{'gap_days':30,'rule':'new period iff consecutive gap >30 days','history_cap':100},
        'mean_groups_per_user':gl.summarize(truncated,users)['groups_per_user']['mean'],
        'mean_interactions_per_group':gl.summarize(truncated,users)['mean_interactions_per_group'],
        'group_length_summary':gl.summarize(truncated,users),'full_observed_history_reference':gl.summarize(full,users),
        'provenance':job['provenance'],'cache_sha256':job['cache_sha256'],'timestamp_verification':verification,
        'mamba_gate':args.mamba_gate,
        'mamba_decay':args.mamba_decay,
        'reusable_intermediates':{'training_influence':diagnostic['model_item_influence_source'],'signals':str(signal_dir),
          'timestamps':str(timestamp_file),'time_group_map':str(group_map),'model_weights_saved':False,
          'supports':['preliminary_statistics','preliminary_grouped_statistics']}}
    base=output/'intermediate/group_lengths.json';save(base,report)
    source={'category':job['name'],'dataset':job['dataset'],'source':str(intermediate),'base_report':str(base)}
    gl.recover=lambda source,cache,selected,data:({u:timestamps[u] for u in selected},verification)
    plot_args=SimpleNamespace(cache_dir=CACHE,output=output,seed=25252,bootstrap=2000,
        min_users=100,device='cuda:0',physical_gpu=args.gpu,styles=STYLES)
    statistics=grouped.calculate(source,plot_args,ROOT)
    grouped.plot({job['name']:statistics},plot_args)
    compact({job['name']:statistics},output)
    final=output/(job['name']+'.json');saved=json.loads(final.read_text())
    saved['mean_groups_per_user']=saved['group_length_summary']['groups_per_user']['mean'];save(final,saved)
    save(output/'status.json',{'status':'complete','gpu':args.gpu,'dataset':job['name'],
        'mamba_gate':args.mamba_gate,'mamba_decay':args.mamba_decay,
        'max_training_epochs':args.epochs,
        'mean_groups_per_user':saved['mean_groups_per_user'],'mean_interactions_per_group':saved['mean_interactions_per_group']})


def bootstrap_tuple(module,values,repetitions,seed):
    if not len(values):return float('nan'),float('nan'),float('nan')
    r=module.gpu_summary(values,seed,repetitions,'cuda:0')
    return r['mean'],float('nan') if r['ci_low'] is None else r['ci_low'],float('nan') if r['ci_high'] is None else r['ci_high']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=list(STYLES));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--mamba-gate',type=float,required=True)
    p.add_argument('--mamba-decay',type=float)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not 0.0 <= args.mamba_gate <= 1.0:p.error('--mamba-gate must be in [0, 1]')
    if args.mamba_decay is not None and not 0.0 <= args.mamba_decay <= 1.0:
        p.error('--mamba-decay must be in [0, 1]')
    save(args.output/'status.json',{'status':'preparing','gpu':args.gpu,'dataset':args.dataset,
                                    'mamba_gate':args.mamba_gate,'mamba_decay':args.mamba_decay})
    try:
        job=prepare(args.dataset,args.output)
        save(args.output/'status.json',{'status':'training','gpu':args.gpu,'dataset':args.dataset,
                                        'epochs':args.epochs,'mamba_gate':args.mamba_gate,
                                        'mamba_decay':args.mamba_decay})
        train(job,args.output/'intermediate/job.json',args.output,args.epochs,
              args.mamba_gate,args.mamba_decay)
        save(args.output/'status.json',{'status':'analyzing','gpu':args.gpu,
                                        'dataset':args.dataset,'mamba_gate':args.mamba_gate,
                                        'mamba_decay':args.mamba_decay})
        analyze(job,args.output,args)
    except Exception as error:
        save(args.output/'status.json',{'status':'failed','dataset':args.dataset,'error':repr(error)})
        raise


if __name__=='__main__':main()
