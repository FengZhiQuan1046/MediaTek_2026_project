"""Redraw saved preliminary statistics with period labels; no training."""
from pathlib import Path
from types import SimpleNamespace
import json,runpy,sys,shutil
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import plot_grouped_preliminary as g
import additional_datasets as a
amazon=ROOT/'outputs/grouped_preliminary'
reports={n:json.loads((amazon/(n+'.json')).read_text())['five_figure_statistics'] for n in ['All_Beauty','Baby_Products','Sports_and_Outdoors','Toys_and_Games']}
g.plot(reports,SimpleNamespace(output=amazon,min_users=100))
runpy.run_path(str(ROOT/'plot_grouped_row.py'))
base=ROOT/'outputs/additional_dated_20261023'
reports={n:json.loads((base/n/(n+'.json')).read_text())['five_figure_statistics'] for n in a.STYLES}
for n,r in reports.items():
 g.plot({n:r},SimpleNamespace(output=base/n,min_users=100,styles=a.STYLES))
 a.compact({n:r},base/n)
g.plot(reports,SimpleNamespace(output=base/'aggregated',min_users=100,styles=a.STYLES))
a.compact(reports,base/'aggregated')
sys.argv=['plot_group_count_overview.py',str(base)]
runpy.run_path(str(ROOT/'plot_group_count_overview.py'))
for src,dst in [(amazon/'figure0_acd_row.pdf','preliminary_grouped_amazon.pdf'),(base/'aggregated/figure0_acd_row.pdf','preliminary_grouped_additional.pdf')]:
 shutil.copy2(src,ROOT.parent/'ver5/Figures'/dst)
print('COMPLETE',flush=True)
