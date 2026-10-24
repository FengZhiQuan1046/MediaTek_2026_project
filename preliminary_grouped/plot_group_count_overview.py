"""Render the already computed average true-time groups/user, no model run."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]);out=root/'aggregated';out.mkdir(exist_ok=True)
rows=[]
for name in ['Steam','MovieLens1M','Yelp19','Yelp23']:
 r=json.loads((root/name/'group_count_summary.json').read_text())
 s=r['sampled_preliminary_cohort_capped']
 rows.append({'dataset':name,**s,'dated_match_fraction':r['provenance'].get('uniquely_dated_fraction',1.0)})
(out/'group_count_overview.json').write_text(json.dumps(rows,indent=2)+'\n')
with (out/'group_count_overview.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'Liberation Serif','pdf.fonttype':42,'svg.fonttype':'path'})
fig,ax=plt.subplots(figsize=(6.8,4))
means=[r['mean_groups_per_user'] for r in rows]
ax.bar([r['dataset'] for r in rows],means,color=['#6A3D9A','#8C564B','#E69F00','#4D4D4D'])
for i,mean in enumerate(means):ax.text(i,mean+.13,f'{mean:.3f}',ha='center')
ax.set_ylabel('Mean temporal periods per user')
ax.set_ylim(0,max(means)*1.16)
ax.set_title('Consecutive gap >30 days starts a new period')
fig.text(.5,.015,'Sampled preliminary cohort · test observed history, max100 items\nYelp19/23 use uniquely dated matched subsets (84.43% /96.77% of positive source rows)',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.08,1,1))
for ext in ['png','pdf','svg']:fig.savefig(out/('mean_groups_per_user.'+ext),dpi=300,bbox_inches='tight')
print('Saved mean-groups chart and JSON/CSV:',out)
