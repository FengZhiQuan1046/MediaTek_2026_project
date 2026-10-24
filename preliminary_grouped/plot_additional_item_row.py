"""Four-panel item-lag diagnostics from cached signals; no model training."""
import ast
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from additional_datasets import STYLES
from period_plot_style import LINESTYLES
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/additional_dated_20261023'
OUT=BASE/'item_level';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('original',ROOT.parent/'preliminary/aggregated/plot_aggregated.py')
style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
# Reuse the original bootstrap implementation exactly, without importing training dependencies.
tree=ast.parse((ROOT.parent/'preliminary/experiment.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bootstrap_mean')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_bootstrap','exec'))
tables={n:[] for n in style.FILES};provenance=[];datasets=[]
for name,(label,color,marker) in STYLES.items():
 folder=BASE/name;job=json.loads((folder/'intermediate/job.json').read_text());ds=job['dataset']
 datasets.append((name,ds,label,color,marker,LINESTYLES[name]))
 for filename in ['alignment_summary.csv','information_coverage.csv']:
  with (folder/'intermediate/signals'/filename).open() as f:tables[filename].extend(csv.DictReader(f))
 source=folder/'intermediate/training'/ds.replace(':','_')/'history_item_influence.csv'
 grouped=defaultdict(list)
 with source.open() as f:
  for row in csv.DictReader(f):
   if row['split']=='test':grouped[int(row['lag'])].append(float(row['margin_influence']))
 for lag,values in sorted(grouped.items()):
  mean,lo,hi=bootstrap_mean(values,2000,25252+lag)
  tables['model_item_influence_summary.csv'].append(dict(dataset=ds,split='test',lag=lag,mean_margin_influence=mean,ci_low=lo,ci_high=hi,n_users=len(values)))
 provenance.append({'dataset':name,'job':str(folder/'intermediate/job.json'),'signals':str(folder/'intermediate/signals'),'influence':str(source)})
for filename,rows in tables.items():
 with (OUT/filename).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
style.DATASETS=datasets
style.style(None)
plt=style.plt
fig,axes=plt.subplots(1,4,figsize=(8.80,2.50))
specs=[('alignment','(a) Semantic','Excess cosine (z)','Lag bin'),('influence','(b) Influence','Margin change','Lag'),('coverage','(c) Coverage','Signal (%)','Recent items'),('preference','(d) Preference match','Excess match (pp)','Lag bin')]
for ax,(kind,title,ylabel,xlabel) in zip(axes,specs):
 style.decorate(ax,title,ylabel,xlabel,kind!='coverage')
 style.draw(ax,tables,kind,100,preference_count=32)
 if kind in ('alignment','preference'):
  ax.set_xticks([1,24.5,48.5,82.5],labels=['1','17–32','33–64','65–100'])
 elif kind=='influence':
  ax.set_xticks([1,20,40,60,80,100]);ax.set_xlim(.5,100.5)
 else:
  coverage_end=max(int(r['recent_items']) for r in tables['information_coverage.csv'] if r['split']=='test')
  ax.set_xlim(.5,coverage_end+.5);ax.set_xticks([1,20,40,60,80,100])
 ax.title.set_fontsize(8);ax.xaxis.label.set_size(7.5);ax.yaxis.label.set_size(7.5);ax.tick_params(labelsize=6.5)
handles=[style.Line2D([],[],color=c,marker=m,linestyle=ls,lw=1.6,markersize=4,markerfacecolor='white',markeredgewidth=.65,label=label) for _,_,label,c,m,ls in datasets]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.998),ncol=4,frameon=False,columnspacing=1.6,handlelength=2.5)
fig.subplots_adjust(left=.075,right=.955,top=.77,bottom=.21,wspace=.28)
pos=axes[3].get_position();axes[3].set_position([pos.x0+.029,pos.y0,pos.width,pos.height])
left=axes[2].get_position().x1;right=axes[3].get_position().x0
fig.add_artist(style.Line2D([left+.30*(right-left)]*2,[.15,.82],transform=fig.transFigure,color='#4B5563',lw=1.55))
style.save(fig,OUT,'figure0_abcd_row',['png','pdf','svg'],400);plt.close(fig)
(OUT/'sources.json').write_text(json.dumps({'sources':provenance,'bootstrap':2000,'seed':25252,'min_users':100,'preference_metric':'history_normalized_cluster_match','aggregation':'Original item lag bins for alignment/preference; individual item lag for influence; recent item count for coverage'},indent=2))
import shutil
shutil.copy2(OUT/'figure0_abcd_row.pdf',ROOT.parent/'ver5/Figures/preliminary_additional_items_abcd.pdf')
print('COMPLETE',OUT,flush=True)
