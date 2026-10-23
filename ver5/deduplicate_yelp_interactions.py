"""Collapse MAPLE aspect rows into user-item interactions; keep raw data."""
import json
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
seen=set()
count=0
with source.open() as inp, output.with_suffix('.tmp').open('w') as out:
    for line in inp:
        row=json.loads(line)
        count+=1
        key=(row['user_id'],row['item_id'])
        if key not in seen:
            seen.add(key)
            out.write(line)
output.with_suffix('.tmp').replace(output)
manifest={'source':str(source),'source_rows':count,'unique_user_item_interactions':len(seen),'removed_duplicate_rows':count-len(seen),'ordering':'first source occurrence; source release lacks timestamps','output':str(output)}
output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest),flush=True)
