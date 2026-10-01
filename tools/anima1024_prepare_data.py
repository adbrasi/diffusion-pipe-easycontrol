#!/usr/bin/env python3
"""Materialize every complete source pair for a fresh Anima A/aligned run.

No content filters, subset quotas, or disk-based truncation. Captions use the
E2 full/short/short recipe; the original caption is retained for every pair.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from krea2_prepare_budget_data import heldout_keys, inventory
from nextscene_captions import action_caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/workspace/ds'))
    parser.add_argument('--heldout-manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    pairs, excluded, hashes = inventory(args.source, heldout_keys(args.heldout_manifest))
    subsets = Counter(p['subset'] for p in pairs)
    if len(subsets) != 4 or not pairs:
        raise RuntimeError(f'Expected four populated subsets, found {dict(subsets)}')
    if args.destination.exists():
        raise FileExistsError(f'Refusing to reuse data/cache: {args.destination}')
    for field in ('target', 'control'):
        (args.destination / field).mkdir(parents=True)
    captions = {}
    short_counts = Counter()
    for row in pairs:
        name = row['output_filename']
        for field in ('target', 'control'):
            os.link(row[field], args.destination / field / name)
        short = action_caption(row['caption'], same_subject=True)
        captions[name] = [row['caption']] + ([short, short] if short else [])
        short_counts[row['subset']] += bool(short)
    (args.destination / 'target/captions.json').write_text(
        json.dumps(captions, ensure_ascii=False, indent=2))
    report = dict(source=str(args.source), destination=str(args.destination),
                  complete_pairs=len(pairs), pairs_by_subset=dict(subsets),
                  short_tier_pairs_by_subset=dict(short_counts),
                  caption_exposures=sum(map(len, captions.values())),
                  technical_exclusions=excluded, source_sha256=hashes,
                  selection='all complete pairs; no semantic filters or sample cap',
                  subset_repeats=1, captions='original full + short + short where extractable',
                  heldout_manifest=str(args.heldout_manifest), pairs=pairs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    summary = {k: v for k, v in report.items() if k != 'pairs'}
    args.report.with_name('dataset_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
