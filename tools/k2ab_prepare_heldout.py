"""Select neutral held-out prompts; keep the difficult identity/environment cases."""
import json
import math
import os
from pathlib import Path

from PIL import Image
from tools.k2ab_prepare_data import DENY, safe

root = Path('/workspace/k2ab')
source = Path('/workspace/heldout')
rows = json.loads(Path('/workspace/nextscene_artifacts/data/heldout_short_manifest.json').read_text())
priority = [0, 3, 9, 7, 1, 11, 10, 15, 19, 23, 8, 5, 17, 21]
selected = []
for index in priority:
    row = rows[index]
    stem = row['heldout_stem']
    a = next((source / 'control').glob(stem + '.*'))
    b = next((source / 'target').glob(stem + '*.jpg'))
    # Both columns in heldout_neutral_review.jpg were visually checked.
    # The rating model falsely rejects several neutral dark anime frames.
    if DENY.search(row['caption']):
        continue
    with Image.open(b) as image:
        scale = math.sqrt(1024 ** 2 / (image.width * image.height))
        width = round(image.width * scale / 16) * 16
        height = round(image.height * scale / 16) * 16
    for field, path in [('control', a), ('target', b)]:
        folder = root / 'heldout' / field
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / path.name
        if not dst.exists():
            os.link(path, dst)
    prompt = row['short_prompt']
    (root / 'heldout/target' / (stem + '.txt')).write_text(prompt)
    selected.append(dict(stem=stem, prompt=prompt, reference=a.name,
                         width=width, height=height, seed=76, subset=row['subset'],
                         visual_review='neutral_clothed', automatic_safe=safe(a) and safe(b)))
for index, row in enumerate(selected):
    row['shuffled_reference'] = selected[(index + len(selected) // 2 + 1) % len(selected)]['reference']
(root / 'artifacts/heldout_manifest.json').write_text(json.dumps(selected, indent=2))
print('Held-out:', len(selected), [row['stem'] for row in selected])
