#!/usr/bin/env python3
"""Plan native Krea2 pairs from original datasets, limited only by cache disk.

Default is a CPU-only dry run. --materialize writes a fresh hardlinked dataset;
it never builds embeddings, runs training, deletes data, or queues GPU jobs.
Captions are copied verbatim. There are no content filters or per-subset quotas.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

from PIL import Image

GIB = 1024 ** 3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def heldout_keys(manifest):
    if manifest is None:
        return set()
    return {(row['subset'], Path(row['target']).name)
            for row in json.loads(manifest.read_text())}


def inventory(source, excluded):
    pairs, rejected, source_hashes = [], Counter(), {}
    for subset in sorted(source.glob('ds*')):
        caption_file = subset / 'captions.jsonl'
        if not (subset / 'images_B').is_dir() or not (subset / 'images_A').is_dir():
            continue
        captions = {}
        if caption_file.is_file():
            source_hashes[str(caption_file)] = sha256(caption_file)
        for line in caption_file.read_text().splitlines() if caption_file.is_file() else []:
            if not line.strip():
                continue
            row = json.loads(line)
            caption = row.get('caption')
            if not isinstance(caption, str) or not caption.strip() or row.get('error'):
                continue
            # Last successful caption revision for this target, without edits.
            filename = Path(row['video']).name
            captions[filename] = caption
        targets = {p.name: p for p in (subset / 'images_B').iterdir()
                   if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')}
        controls = {}
        for p in (subset / 'images_A').iterdir():
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                controls.setdefault(p.stem, []).append(p)
        caption_digest = hashlib.sha256()
        for filename, target in sorted(targets.items()):
            control = subset / 'images_A' / filename
            if (subset.name, filename) in excluded:
                rejected['heldout_pairs'] += 1
                continue
            # Published .txt captions include successful retries absent from raw
            # JSONL generation logs, and are the dataset's authoritative captions.
            published_caption = target.with_suffix('.txt')
            caption = published_caption.read_text() if published_caption.is_file() else captions.get(filename, '')
            if not caption.strip():
                rejected['target_without_caption'] += 1
                continue
            if not control.is_file():
                matches = controls.get(target.stem, [])
                if len(matches) != 1:
                    rejected['missing_or_ambiguous_scene1'] += 1
                    continue
                control = matches[0]
            try:
                for image_path in (control, target):
                    with Image.open(image_path) as image:
                        if min(image.size) < 1:
                            raise ValueError('Empty image')
                        image.verify()
            except (OSError, ValueError, SyntaxError):
                rejected['unreadable_image_pair'] += 1
                continue
            # Explicit source prefix prevents collisions across subsets.
            caption_digest.update(json.dumps([filename, caption], ensure_ascii=False).encode())
            pairs.append(dict(subset=subset.name, filename=filename,
                              output_filename=f'{subset.name}__{filename}',
                              target=str(target), control=str(control),
                              caption=caption))
        source_hashes[str(subset / 'images_B/*.txt (eligible caption digest)')] = caption_digest.hexdigest()
    return pairs, dict(rejected), source_hashes


def select_pairs(pairs, free_bytes, reserve_bytes, pair_bytes, overhead, seed):
    if pair_bytes <= 0 or overhead < 1 or reserve_bytes < 0:
        raise ValueError('Invalid budget settings')
    budget = max(0, free_bytes - reserve_bytes)
    capacity = math.floor(budget / (pair_bytes * overhead))
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[:capacity], capacity


def materialize(pairs, destination):
    if destination.exists():
        raise FileExistsError(f'Refusing to reuse a dataset tree: {destination}')
    destination.mkdir(parents=True)
    for field in ('target', 'control'):
        (destination / field).mkdir()
    captions = {}
    for row in pairs:
        name = row['output_filename']
        for field in ('target', 'control'):
            os.link(row[field], destination / field / name)
        captions[name] = [row['caption']]
    (destination / 'target/captions.json').write_text(json.dumps(captions, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/workspace/ds'))
    parser.add_argument('--heldout-manifest', type=Path,
                        default=Path('/workspace/nextscene_artifacts/data/heldout_short_manifest.json'))
    parser.add_argument('--destination', type=Path, default=Path('/workspace/k2ab/data_fullbudget_20261001'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reserve-gib', type=float, default=8)
    parser.add_argument('--pair-mb', type=float, default=24.8,
                        help='Decimal MB per pair, before overhead (estimate; measure during cache smoke).')
    parser.add_argument('--cache-overhead', type=float, default=1.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--materialize', action='store_true')
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error('Source dataset directory missing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates, rejected, hashes = inventory(args.source, heldout_keys(args.heldout_manifest))
    free = shutil.disk_usage(args.output.parent).free
    pair_bytes = args.pair_mb * 1_000_000
    reserve = args.reserve_gib * GIB
    selected, capacity = select_pairs(candidates, free, reserve, pair_bytes, args.cache_overhead, args.seed)
    if not selected:
        raise RuntimeError('No complete pairs fit the cache budget')
    report = dict(status='planned', seed=args.seed, source=str(args.source),
                  destination=str(args.destination), eligible_pairs=len(candidates),
                  selected_pairs=len(selected), capacity_pairs=capacity,
                  eligible_by_subset=dict(Counter(p['subset'] for p in candidates)),
                  selected_by_subset=dict(Counter(p['subset'] for p in selected)),
                  technical_exclusions=rejected, caption_source_sha256=hashes,
                  free_bytes=free, reserve_bytes=reserve,
                  estimated_bytes_per_pair=pair_bytes, cache_overhead=args.cache_overhead,
                  estimated_cache_bytes=len(selected) * pair_bytes,
                  budgeted_cache_bytes=len(selected) * pair_bytes * args.cache_overhead,
                  selection_rule='seeded global shuffle, no content filters or subset quotas',
                  estimate_warning='Measure actual native cache bytes/pair before building the complete cache.',
                  captions='original strings, one caption per pair, no rewriting',
                  pairs=selected)
    if args.materialize:
        materialize(selected, args.destination)
        report['status'] = 'materialized_without_cache'
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
