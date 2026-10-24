"""Reusable per-user true-time group counts, independent of model training."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import pickle
import sys

import duckdb
import numpy as np
import pyarrow as pa

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ver4'))


def calculate(output):
    job=json.loads((output/'intermediate/job.json').read_text())
    with Path(job['cache_path']).open('rb') as stream:data=pickle.load(stream)
    con=duckdb.connect()
    con.execute('SET threads=4')
    path=job['filtered_timestamps'].replace("'","''")
    con.execute(f"""CREATE TABLE observed AS SELECT * FROM (
      SELECT user_id,time_ms,source_order,
        row_number() OVER(PARTITION BY user_id ORDER BY time_ms DESC,source_order DESC) rev,
        count(*) OVER(PARTITION BY user_id) n FROM read_parquet('{path}')
      ) WHERE n>=3 AND rev>=2""")
    def counts(where):
        return con.execute(f"""SELECT user_id,count(*) interactions,
          sum(CASE WHEN prev IS NULL OR time_ms-prev>2592000000 THEN 1 ELSE 0 END)::BIGINT group_count
          FROM (SELECT *,lag(time_ms) OVER(PARTITION BY user_id ORDER BY time_ms,source_order) prev
                FROM observed {where}) GROUP BY user_id ORDER BY user_id""").fetchall()
    capped=counts('WHERE rev<=101');full=counts('')
    assert len(capped)==len(data.test_target)
    all_user_names=con.execute(f"SELECT DISTINCT user_id FROM read_parquet('{path}') ORDER BY user_id").fetchnumpy()['user_id']
    users=np.array(sorted(set(data.valid_target)&set(data.test_target)))
    if len(users)>2000:users=np.sort(np.random.default_rng(25252).choice(users,2000,replace=False))
    sampled=set(all_user_names[u] for u in users)
    def summary(rows):
        return {'users':len(rows),'mean_groups_per_user':float(np.mean([r[2] for r in rows])),
            'median_groups_per_user':float(np.median([r[2] for r in rows])),
            'mean_interactions_per_group':sum(r[1] for r in rows)/sum(r[2] for r in rows),
            'total_groups':sum(r[2] for r in rows),'total_observed_interactions':sum(r[1] for r in rows)}
    report={'dataset':job['name'],'scope':'test observed history=train+validation, test target excluded',
      'gap_days':30,'boundary':'new group only when consecutive gap >30 days','history_cap':100,
      'all_eligible_users_capped':summary(capped),
      'sampled_preliminary_cohort_capped':summary([r for r in capped if r[0] in sampled]),
      'all_eligible_users_full_history':summary(full),'provenance':job['provenance']}
    (output/'group_count_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    full_map={r[0]:r for r in full}
    with gzip.open(output/'intermediate/per_user_group_counts.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['original_user_id','in_sampled_cohort','capped_interactions','capped_groups','full_interactions','full_groups'])
        for user,interactions,groups in capped:
            f=full_map[user];writer.writerow([user,user in sampled,interactions,groups,f[1],f[2]])
    print(job['name'],report['sampled_preliminary_cohort_capped'],flush=True)
    con.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    args=p.parse_args()
    for name in ['Steam','MovieLens1M','Yelp19','Yelp23']:calculate(args.root/name)
