#!/usr/bin/env python3
"""Audit and filter next-scene pairs before training (anima_nextscene).

Why: the two recurring failures are the CAPTION shortcut (text describes the
whole target) and the COPY shortcut (reference ~= target, so reconstructing it
is a low-loss solution). This tool measures the second one per pair and builds
a filtered training tree; it also reports caption length so the first one is
visible before any GPU time is spent.

Layout in (diffusion-pipe edit dataset):
    <target_dir>/<stem>.<img>  + <stem>.txt      (scene 2 = what the model generates)
    <control_dir>/<stem>.<img>                    (scene 1 = reference)

Per pair it computes:
    dhash_ham   64-bit difference-hash Hamming distance (0 = identical layout)
    pix_sim     cosine similarity of 32x32 grayscale thumbnails (layout/colour copy proxy)
    dino_cos    optional (--dino): DINOv2 CLS cosine (semantic relatedness)
    words       caption word count

Buckets (thresholds are starting points — tune them by LOOKING at the report):
    near_dup    dhash_ham <= --min-dhash  or  pix_sim >= --max-pix-sim   -> dropped (pure copy lesson)
    unrelated   dino_cos < --min-dino (only with --dino)                  -> dropped (cut / different show)
    keep        everything else

Usage:
    python tools/nextscene_pairs.py audit  --target DIR --control DIR --out report.csv [--dino]
    python tools/nextscene_pairs.py build  --report report.csv --out-root /workspace/ns_filtered \\
        [--reverse]   # also emit B->A pairs when the reference has its own .txt caption

`build` hardlinks (falls back to copy) into <out-root>/target and <out-root>/control,
so the filtered tree costs no disk on the same filesystem.
"""

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def find_images(d):
    return {p.stem: p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT}


def dhash(img, size=8):
    g = img.convert('L').resize((size + 1, size), Image.BILINEAR)
    px = list(g.tobytes())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (row[c] > row[c + 1])
    return bits


def thumb_vec(img, size=32):
    g = img.convert('L').resize((size, size), Image.BILINEAR)
    v = [x / 255.0 for x in g.tobytes()]
    m = sum(v) / len(v)
    return [x - m for x in v]


def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b)) or 1e-8
    return num / den


class Dino:
    def __init__(self, name='facebook/dinov2-small'):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.dev).eval()

    def cos(self, a, b):
        t = self.torch
        with t.no_grad():
            x = self.proc(images=[a.convert('RGB'), b.convert('RGB')], return_tensors='pt').to(self.dev)
            f = self.model(**x).last_hidden_state[:, 0]
            f = t.nn.functional.normalize(f, dim=-1)
        return float((f[0] * f[1]).sum())


def audit(args):
    targets = find_images(args.target)
    controls = find_images(args.control)
    stems = sorted(set(targets) & set(controls))
    missing = len(targets) - len(stems)
    dino = Dino() if args.dino else None
    rows = []
    for i, stem in enumerate(stems):
        tp, cp = targets[stem], controls[stem]
        try:
            a, b = Image.open(cp), Image.open(tp)
            a.load(); b.load()
        except Exception as e:  # noqa: BLE001
            print(f'skip {stem}: {e}', file=sys.stderr)
            continue
        cap_file = tp.with_suffix('.txt')
        caption = cap_file.read_text().strip() if cap_file.exists() else ''
        ref_cap = cp.with_suffix('.txt')
        row = {
            'stem': stem, 'target': str(tp), 'control': str(cp),
            'target_caption': str(cap_file) if cap_file.exists() else '',
            'control_caption': str(ref_cap) if ref_cap.exists() else '',
            'dhash_ham': bin(dhash(a) ^ dhash(b)).count('1'),
            'pix_sim': round(cos(thumb_vec(a), thumb_vec(b)), 4),
            'dino_cos': round(dino.cos(a, b), 4) if dino else '',
            'words': len(caption.split()),
            'size_match': int(a.size == b.size),
        }
        near_dup = row['dhash_ham'] <= args.min_dhash or row['pix_sim'] >= args.max_pix_sim
        unrelated = dino is not None and row['dino_cos'] < args.min_dino
        row['bucket'] = 'near_dup' if near_dup else ('unrelated' if unrelated else 'keep')
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f'{i + 1}/{len(stems)}', file=sys.stderr)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    counts = {k: sum(r['bucket'] == k for r in rows) for k in ('keep', 'near_dup', 'unrelated')}
    words = sorted(r['words'] for r in rows)
    print(f'pairs: {n}  (targets without control: {missing})')
    for k, v in counts.items():
        print(f'  {k:10s} {v:7d}  ({100 * v / max(n, 1):.1f}%)')
    if words:
        print(f'caption words: median {words[n // 2]}  p90 {words[int(n * 0.9)]}  '
              f'(>60 words = caption-shortcut risk: the text alone can reconstruct B)')
    print(f'size mismatch A/B: {sum(1 - r["size_match"] for r in rows)} pairs '
          f'(fine — both are fit to the same bucket — but check the crops look sane)')
    print(f'report: {args.out}')


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(args):
    out = Path(args.out_root)
    kept = rev = 0
    with open(args.report) as f:
        for r in csv.DictReader(f):
            if r['bucket'] != 'keep':
                continue
            tp, cp = Path(r['target']), Path(r['control'])
            _link(tp, out / 'target' / tp.name)
            if r['target_caption']:
                _link(Path(r['target_caption']), out / 'target' / (tp.stem + '.txt'))
            _link(cp, out / 'control' / (tp.stem + cp.suffix))
            kept += 1
            # Next-scene has no privileged direction: B->A is a free second pair
            # when A has its own caption (the April bilingual dataset did this).
            if args.reverse and r['control_caption']:
                stem = f'{tp.stem}__rev'
                _link(cp, out / 'target' / (stem + cp.suffix))
                _link(Path(r['control_caption']), out / 'target' / (stem + '.txt'))
                _link(tp, out / 'control' / (stem + tp.suffix))
                rev += 1
    print(f'built {kept} pairs (+{rev} reversed) in {out}/target + {out}/control')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--target', required=True)
    a.add_argument('--control', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--dino', action='store_true', help='also compute DINOv2 relatedness (needs transformers)')
    a.add_argument('--min-dhash', type=int, default=6, help='<= this Hamming distance (of 64) = near-duplicate')
    a.add_argument('--max-pix-sim', type=float, default=0.97, help='>= this thumbnail cosine = near-duplicate')
    a.add_argument('--min-dino', type=float, default=0.35, help='< this DINO cosine = unrelated (with --dino)')
    b = sub.add_parser('build')
    b.add_argument('--report', required=True)
    b.add_argument('--out-root', required=True)
    b.add_argument('--reverse', action='store_true')
    args = ap.parse_args()
    audit(args) if args.cmd == 'audit' else build(args)


if __name__ == '__main__':
    main()
