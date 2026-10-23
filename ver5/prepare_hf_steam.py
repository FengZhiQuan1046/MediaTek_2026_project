"""Normalize the HF Steam release in external cache; preserve original files."""
import ast
import csv
import gzip
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
output=Path(sys.argv[2])
texts={}
with gzip.open(root/'steam_games.json.gz','rt',encoding='utf-8') as f:
    for line in f:
        row=ast.literal_eval(line)
        if row.get('id'):
            texts[str(row['id'])]=' '.join(str(x) for x in [row.get('title') or row.get('app_name') or row['id'], 'Genres: '+', '.join(row.get('genres') or []), 'Tags: '+', '.join((row.get('tags') or [])[:15])])
output.parent.mkdir(parents=True,exist_ok=True)
count=0
with gzip.open(root/'steam_reviews.json.gz','rt',encoding='utf-8') as f, output.with_suffix('.tmp').open('w',newline='',encoding='utf-8') as out:
    writer=csv.writer(out)
    writer.writerow(['user_id','item_id','date','item_text'])
    for line in f:
        row=ast.literal_eval(line)
        if not row.get('username') or not row.get('product_id') or not row.get('date'):
            raise ValueError('Steam row lacks user, product or date')
        item=str(row['product_id'])
        writer.writerow([row['username'],item,row['date'],texts.get(item,item)])
        count+=1
        if count%1000000==0: print('Steam rows',count,flush=True)
output.with_suffix('.tmp').replace(output)
manifest={'source':'recommender-system/steam-review-and-bundle-dataset','rows':count,'input':str(root),'output':str(output),'semantics':'All observed reviews are implicit interactions; no explicit positive/negative label in source. Item metadata only; review text excluded.'}
output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest),flush=True)
