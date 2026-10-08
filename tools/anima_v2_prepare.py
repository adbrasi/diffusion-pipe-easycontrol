#!/usr/bin/env python3
"""Materialize a new paired dataset for the Anima A/aligned 1024 run (v2).

Source layout (same as proxima_cena_grounded_original_dataset): <source>/ds*/images_A
(reference) + images_B (target + <stem>.txt). Any number of ds* subsets.

Held-out is chosen BEFORE training data is written: --per-subset pairs per subset
(fixed seed), and every pair sharing the held-out pair's video id (`<id>_image_NNNN`
stems) is excluded from training, so neighbouring frames cannot leak.
Training captions follow the E2 recipe: original full caption + two copies of the
extracted short action caption when one exists. No content filters or sample caps.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil

from PIL import Image

from krea2_prepare_budget_data import inventory
from nextscene_captions import action_caption


def group_key(row):
    # Same pair under several caption subsets shares one key, so held-out is leak-free across them.
    stem = Path(row['filename']).stem
    return stem.split('_image_')[0] if '_image_' in stem else stem


def stratum(row, by_prefix):
    return row['filename'].split('_')[0] if by_prefix else row['subset']


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--heldout', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--per-subset', type=int, default=6)
    parser.add_argument('--seed', type=int, default=76)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epochs', type=float, default=5)
    parser.add_argument('--no-short', action='store_true', help='Use each caption verbatim (no extracted short tiers)')
    parser.add_argument('--stratify-prefix', action='store_true',
                        help='Balance held-out by filename source prefix (ds1_, ds2_, ...) instead of subset folder')
    parser.add_argument('--eval-subset', default=None, help='Subset whose caption becomes the held-out eval prompt')
    args = parser.parse_args()
    for path in (args.destination, args.heldout):
        if path.exists():
            raise FileExistsError(f'Refusing to reuse {path}')

    pairs, excluded, hashes = inventory(args.source, set())
    if not pairs:
        raise RuntimeError(f'No complete pairs under {args.source}/ds*/images_{{A,B}}')
    groups = defaultdict(list)
    for row in pairs:
        groups[group_key(row)].append(row)

    rng = random.Random(args.seed)
    heldout_by_subset = defaultdict(list)
    heldout_groups = set()
    eval_rows = [r for r in pairs if args.eval_subset in (None, r['subset'])]
    for name in sorted({stratum(r, args.stratify_prefix) for r in eval_rows}):
        keys = sorted({group_key(r) for r in eval_rows if stratum(r, args.stratify_prefix) == name})
        for key in rng.sample(keys, min(args.per_subset, len(keys))):
            heldout_groups.add(key)
            heldout_by_subset[name].append(rng.choice([r for r in groups[key] if r in eval_rows]))
    train = [r for r in pairs if group_key(r) not in heldout_groups]
    leak_excluded = len(pairs) - len(train) - sum(map(len, heldout_by_subset.values()))

    # Held-out: round-robin across subsets so the first N eval rows cover every subset.
    for field in ('target', 'control'):
        (args.heldout / field).mkdir(parents=True)
    heldout_rows, index = [], 0
    queues = [list(v) for _, v in sorted(heldout_by_subset.items())]
    while any(queues):
        for queue in queues:
            if not queue:
                continue
            row = queue.pop(0)
            stem = f'{index:02d}_{Path(row["filename"]).stem}'
            target = args.heldout / 'target' / (stem + Path(row['target']).suffix.lower())
            control = args.heldout / 'control' / (stem + Path(row['control']).suffix.lower())
            link(row['target'], target)
            link(row['control'], control)
            prompt = row['caption'] if args.no_short else (action_caption(row['caption'], same_subject=True) or row['caption'])
            target.with_suffix('.txt').write_text(prompt)
            heldout_rows.append(dict(row, eval_name=stem, eval_prompt=prompt))
            index += 1
    (args.heldout / 'manifest.json').write_text(json.dumps(heldout_rows, ensure_ascii=False, indent=2))

    for field in ('target', 'control'):
        (args.destination / field).mkdir(parents=True)
    captions, short_counts, ars = {}, Counter(), []
    for row in train:
        name = row['output_filename']
        for field in ('target', 'control'):
            link(row[field], args.destination / field / name)
        short = None if args.no_short else action_caption(row['caption'], same_subject=True)
        captions[name] = [row['caption']] + ([short, short] if short else [])
        short_counts[row['subset']] += bool(short)
        with Image.open(row['target']) as im:
            ars.append((im.width / im.height, len(captions[name])))
    (args.destination / 'target/captions.json').write_text(json.dumps(captions, ensure_ascii=False, indent=2))

    # Estimate steps/epoch with the loader's 7 AR buckets (0.5-2.0) and pad_last_batch.
    import numpy as np
    buckets = np.geomspace(0.5, 2.0, num=7)
    per_bucket = Counter()
    for ar, n in ars:
        per_bucket[int(np.argmin(np.abs(np.log(buckets) - math.log(ar))))] += n
    steps_per_epoch = sum(math.ceil(n / args.batch) for n in per_bucket.values())
    report = dict(source=str(args.source), destination=str(args.destination), heldout=str(args.heldout),
                  complete_pairs=len(pairs), train_pairs=len(train),
                  train_pairs_by_subset=dict(Counter(r['subset'] for r in train)),
                  heldout_pairs=len(heldout_rows), heldout_neighbour_exclusions=leak_excluded,
                  short_tier_pairs_by_subset=dict(short_counts),
                  caption_exposures=sum(map(len, captions.values())),
                  technical_exclusions=excluded, source_sha256=hashes,
                  estimated_steps_per_epoch=steps_per_epoch,
                  estimated_max_steps=int(steps_per_epoch * args.epochs),
                  bucket_presentations={f'{buckets[k]:.3f}': v for k, v in sorted(per_bucket.items())},
                  selection='all complete pairs; no semantic filters or sample cap', subset_repeats=1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(report, pairs=train), ensure_ascii=False, indent=2))
    summary = {k: v for k, v in report.items() if k != 'source_sha256'}
    args.report.with_name('dataset_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
