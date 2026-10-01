#!/usr/bin/env python3
"""Inventory every original complete pair for native DiffSynth Qwen 2.1 Edit."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from krea2_prepare_budget_data import inventory, heldout_keys

def main():
    out = Path('/workspace/qwen21')
    heldout = Path('/workspace/nextscene_artifacts/data/heldout_short_manifest.json')
    pairs, rejected, hashes = inventory(Path('/workspace/ds'), heldout_keys(heldout))
    if len(pairs) != 11526:
        raise RuntimeError(f'Unexpected full inventory: {len(pairs)}')
    rows = [dict(id=p['output_filename'], subset=p['subset'], image=p['target'],
                 edit_image=p['control'], prompt=p['caption']) for p in pairs]
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    (out/'dataset.jsonl').write_text(content)
    (out/'dataset_summary.json').write_text(json.dumps(dict(
        pairs=len(rows), by_subset=dict(Counter(p['subset'] for p in pairs)),
        technical_exclusions=rejected, source_sha256=hashes,
        dataset_sha256=hashlib.sha256(content.encode()).hexdigest(),
        epochs=3, presentations=3*len(rows), repeat=1,
        caption_policy='one original unmodified caption per pair',
        pairing='image=scene B; edit_image=scene A',
        resolution='1024-area, native aspect ratio, dimensions divisible by 32',
        batch_size=1, gradient_accumulation_steps=1,
        framework='DiffSynth-Studio',
        upstream_commit='974cfa37f27ac55eba3b6d10efa21f876900572d'),indent=2))
    heldout_rows=[]
    for p in json.loads(heldout.read_text()):
        target=Path('/workspace/ds')/p['target']
        controls=list((target.parent.parent/'images_A').glob(target.stem+'.*'))
        controls=[c for c in controls if c.suffix.lower() in ('.jpg','.jpeg','.png','.webp')]
        if len(controls)!=1: raise RuntimeError('Ambiguous heldout reference')
        caption=target.with_suffix('.txt').read_text()
        heldout_rows.append(dict(id=p['heldout_stem'], image=str(target),
                                 edit_image=str(controls[0]), prompt=caption))
    (out/'heldout.json').write_text(json.dumps(heldout_rows,ensure_ascii=False,indent=2))
    print('FULL_DATASET',len(rows), 'EPOCHS',3,'PRESENTATIONS',len(rows)*3)

if __name__=='__main__': main()
