"""Match the original figure0_acd_row style using temporal-group statistics."""
import importlib.util
import argparse
import json
from period_plot_style import MARKER
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_style',HERE.parent/'preliminary/aggregated/plot_aggregated.py')
style=importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
plt=style.plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=HERE/'outputs/grouped_preliminary')
args=parser.parse_args()
out=args.output
required=[out/(name+'.json') for name in
          ['All_Beauty','Baby_Products','Sports_and_Outdoors','Toys_and_Games']]
if not all(path.exists() for path in required):
    print('Skipping Amazon grouped row: required Amazon reports are not present in',out)
    raise SystemExit(0)
style.style(None)
fig,axes=plt.subplots(1,3,figsize=(7.15,2.55))
specs=[('alignment','(a) Semantic alignment','Excess cosine (z)'),
       ('influence','(b) Item influence','Margin change'),
       ('coverage','(c) Signal coverage','Signal covered (%)')]
categories=['All_Beauty','Baby_Products','Sports_and_Outdoors','Toys_and_Games']
for ax,(kind,title,ylabel) in zip(axes,specs):
    style.decorate(ax,title,ylabel,'Time period',kind!='coverage')
    for category,(_,_,name,color,marker,linestyle) in zip(categories,style.DATASETS):
        report=json.loads((out/(category+'.json')).read_text())
        stats=report['five_figure_statistics'][kind]
        rows=stats['all_available'] if kind=='alignment' else stats
        rows=[r for r in rows if kind=='coverage' or r['n_users']>=100]
        if not rows:continue
        x=[r['group'] for r in rows]
        ax.plot(x,[r['mean'] for r in rows],color=color,marker=marker,ls=linestyle,
                markevery=max(1,len(rows)//9),label=name,**MARKER)
        lo,hi=('q25','q75') if kind=='coverage' else ('ci_low','ci_high')
        ci=[r for r in rows if r.get(lo) is not None]
        ax.fill_between([r['group'] for r in ci],[r[lo] for r in ci],[r[hi] for r in ci],
                        color=color,alpha=.08 if kind=='coverage' else .11,linewidth=0,zorder=1)
    if kind=='coverage':
        ax.set_xlim(.5,20.5);ax.set_xticks([1,5,10,15,20])
        ax.set_ylim(0,102);ax.axhline(80,color='#777777',lw=.65,ls='--',zorder=1)
    ax.title.set_fontsize(8.5)
    ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
    ax.tick_params(labelsize=7)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.998),ncol=4,frameon=False,columnspacing=1.6,handlelength=2.5)
fig.subplots_adjust(left=.085,right=.985,top=.78,bottom=.21,wspace=.20)
style.save(fig,out,'figure0_abc_row',['png','pdf','svg'],400)
plt.close(fig)
print('Created grouped figure0_abc_row.png/pdf/svg with panels (a), (b), (c).')
