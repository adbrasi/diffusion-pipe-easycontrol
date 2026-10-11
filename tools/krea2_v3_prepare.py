#!/usr/bin/env python3
"""Materialize the v11 natural-caption next-scene set for the Krea 2 Raw run (v3).

Source: AdwolfCzar/ofificial_next_scene_v11 dataset-2-natural.zip -> images_A (reference),
images_B (target + .txt), same filename. Output layout for diffusion-pipe:
  <dest>/train/{target,control}   <dest>/heldout/{target,control} (+ manifest.json)

* Drops junk captions (< 6 words: 'test', 'Decline', 'TEXT_PLACEHOLDER', ...).
* Held-out: whole video groups (<prefix>_<videoid>_image_NNNN) removed from train, so
  neighbouring frames cannot leak. Stratified by filename prefix (ds1_, ds3_, ...).
* Crop audit with the training AR buckets: how much of B is cropped by its bucket and how much
  of A is cropped when crop-fit to B's bucket (reference_pixels='target'). Pairs whose A would
  lose more than --max-ref-crop are dropped (logged).
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import re

from PIL import Image

AR_BUCKETS = [(1, 2), (9, 16), (2, 3), (3, 4), (1, 1), (4, 3), (3, 2), (16, 9), (2, 1), (7, 3)]


def bucket_ar(ar):
    return min((w / h for w, h in AR_BUCKETS), key=lambda r: abs(math.log(ar / r)))


def crop_frac(src_ar, dst_ar):
    """Fraction of the source area removed by a center crop-fit to dst_ar."""
    return 1 - min(src_ar, dst_ar) / max(src_ar, dst_ar)


def group_key(stem):
    return stem.split('_image_')[0] if '_image_' in stem else stem


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='dir with images_A/ images_B/')
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--heldout', type=int, default=12)
    ap.add_argument('--seed', type=int, default=76)
    ap.add_argument('--min-words', type=int, default=6)
    ap.add_argument('--neighbour-radius', type=int, default=8)
    ap.add_argument('--max-ref-crop', type=float, default=0.2)
    args = ap.parse_args()
    if args.dest.exists():
        raise FileExistsError(args.dest)

    rows, dropped = [], defaultdict(list)
    for target in sorted((args.source / 'images_B').iterdir()):
        if target.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            continue
        control = args.source / 'images_A' / target.name
        txt = target.with_suffix('.txt')
        if not control.exists() or not txt.exists():
            dropped['missing'].append(target.stem)
            continue
        caption = txt.read_text().strip()
        if len(caption.split()) < args.min_words:
            dropped['junk_caption'].append(f'{target.stem}: {caption!r}')
            continue
        with Image.open(target) as b, Image.open(control) as a:
            b_ar, a_ar = b.width / b.height, a.width / a.height
            b_size, a_size = b.size, a.size
        bucket = bucket_ar(b_ar)
        row = dict(stem=target.stem, target=str(target), control=str(control), caption=caption,
                   b_size=b_size, a_size=a_size, bucket=round(bucket, 4),
                   b_crop=round(crop_frac(b_ar, bucket), 4), a_crop=round(crop_frac(a_ar, bucket), 4))
        if row['a_crop'] > args.max_ref_crop:
            dropped['ref_crop'].append(f"{target.stem}: A {a_size} vs B {b_size} -> {row['a_crop']:.0%}")
            continue
        rows.append(row)

    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r['stem'])].append(r)
    rng = random.Random(args.seed)
    by_prefix = defaultdict(list)
    for key in sorted(groups):
        by_prefix[key.split('_')[0]].append(key)
    # Round-robin over prefixes, proportional-ish: at least 1 per prefix.
    held_keys, prefixes = [], sorted(by_prefix)
    pools = {p: rng.sample(by_prefix[p], len(by_prefix[p])) for p in prefixes}
    weights = {p: len(by_prefix[p]) for p in prefixes}
    while len(held_keys) < args.heldout:
        p = max(prefixes, key=lambda q: weights[q] / (1 + sum(k.startswith(q + '_') or k == q for k in held_keys)))
        if not pools[p]:
            weights[p] = 0
            continue
        held_keys.append(pools[p].pop())
    heldout = [rng.choice(groups[k]) for k in held_keys]

    def near_heldout(stem):
        # ds2/ds3/ds4 stems are sequential frame numbers without a video id: drop +-radius too.
        if group_key(stem) in set(held_keys):
            return True
        m = re.match(r'^(.*?)(\d+)$', stem)
        if not m or '_image_' in stem:
            return False
        for h in heldout:
            hm = re.match(r'^(.*?)(\d+)$', h['stem'])
            if hm and hm.group(1) == m.group(1) and abs(int(hm.group(2)) - int(m.group(2))) <= args.neighbour_radius:
                return True
        return False
    train = [r for r in rows if not near_heldout(r['stem'])]

    for split, items in (('train', train), ('heldout', heldout)):
        for field in ('target', 'control'):
            (args.dest / split / field).mkdir(parents=True)
        for i, r in enumerate(items):
            name = r['stem'] if split == 'train' else f'{i:02d}_{r["stem"]}'
            t, c = Path(r['target']), Path(r['control'])
            link(t, args.dest / split / 'target' / (name + t.suffix.lower()))
            link(c, args.dest / split / 'control' / (name + t.suffix.lower() if c.suffix.lower() == t.suffix.lower() else name + c.suffix.lower()))
            (args.dest / split / 'target' / (name + '.txt')).write_text(r['caption'])
            r['name'] = name
    (args.dest / 'heldout' / 'manifest.json').write_text(json.dumps(heldout, indent=1, ensure_ascii=False))
    (args.dest / 'train_manifest.json').write_text(json.dumps(train, indent=1, ensure_ascii=False))

    report = dict(
        source=str(args.source), pairs_total=len(rows) + sum(map(len, dropped.values())),
        train=len(train), heldout=len(heldout), heldout_groups=held_keys,
        excluded_neighbours=len(rows) - len(train) - len(heldout),
        dropped={k: v for k, v in dropped.items()},
        buckets=Counter(f'{r["bucket"]:.3f}' for r in train).most_common(),
        b_crop_mean=sum(r['b_crop'] for r in train) / len(train),
        a_crop_mean=sum(r['a_crop'] for r in train) / len(train),
        a_crop_gt5pct=sum(r['a_crop'] > 0.05 for r in train),
        b_crop_gt5pct=sum(r['b_crop'] > 0.05 for r in train),
        prefixes=Counter(r['stem'].split('_')[0] for r in train),
    )
    (args.dest / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'dropped'}, indent=1))
    print({k: len(v) for k, v in dropped.items()})


if __name__ == '__main__':
    main()
