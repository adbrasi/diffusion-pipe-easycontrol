#!/usr/bin/env python3
"""Generator-agnostic metrics for the Krea 2 A/B (arm A renders in stock ComfyUI,
arm B in the runner) — same numbers as tools/nextscene_eval.py, computed from files.

Layout per checkpoint directory:
    <dir>/<stem>_true.png       output with the correct reference
    <dir>/<stem>_shuffled.png   same prompt/seed, reference of another held-out pair
and the held-out pairs as for training: <pairs>/target/<stem>.* (+ .txt), <pairs>/control/<stem>.*

Metrics (DINOv2 cosine against the REAL next scene B; CCIP if dghs-imgutils is installed):
    gt_true, ref_gain = sim(true,B) - sim(shuffled,B), copy_gap = sim(true,A) - sim(B,A),
    copy_rate (dHash <= 6 between output and A), ccip_true (same character as B).
Writes <dir>/metrics.json and a 4-column grid A | B | true | shuffled.

    python tools/k2ab_metrics.py --pairs /workspace/k2ab/heldout --dir run/A_native/step250 [--dir ...]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.nextscene_eval import Feats, dhash_ham  # noqa: E402

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp'}


def find(d, stem):
    for p in Path(d).glob(f'{stem}.*'):
        if p.suffix.lower() in IMG_EXT:
            return p
    return None


def labeled_grid(rows, stems, title, cell=384):
    header, label = 64, 28
    resized_rows = []
    for row in rows:
        resized = []
        for image in row:
            image = image.copy()
            image.thumbnail((cell, cell))
            resized.append(image)
        resized_rows.append(resized)
    heights = [max(image.height for image in row) for row in resized_rows]
    canvas = Image.new('RGB', (4*cell, header + sum(heights) + len(rows)*label), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    small = ImageFont.truetype('DejaVuSans.ttf', 14)
    draw.text((8, 6), title, fill='black', font=font)
    for column, text in enumerate(('A - referencia', 'B - proxima cena real', 'Ref certa - resultado', 'Ref trocada - resultado')):
        draw.text((column*cell+8, 35), text, fill='black', font=font)
    y = header
    for index, (row, stem, height) in enumerate(zip(resized_rows, stems, heights)):
        draw.text((8, y+4), f'{index+1:02d} | {stem}', fill='black', font=small)
        for column, image in enumerate(row):
            canvas.paste(image, (column*cell+(cell-image.width)//2, y+label+(height-image.height)//2))
        y += height + label
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--dir', action='append', required=True)
    ap.add_argument('--copy_dhash', type=int, default=6)
    args = ap.parse_args()
    feats = Feats('cuda' if torch.cuda.is_available() else 'cpu')
    for d in map(Path, args.dir):
        rows, per = [], []
        for out_true in sorted(d.glob('*_true.png')):
            stem = out_true.name[:-len('_true.png')]
            a_path, b_path = find(Path(args.pairs) / 'control', stem), find(Path(args.pairs) / 'target', stem)
            shuf = d / f'{stem}_shuffled.png'
            if a_path is None or b_path is None or not shuf.exists():
                print(f'skip {stem}: missing A/B/shuffled')
                continue
            t, s = Image.open(out_true).convert('RGB'), Image.open(shuf).convert('RGB')
            A = Image.open(a_path).convert('RGB').resize(t.size)
            B = Image.open(b_path).convert('RGB').resize(t.size)
            f = feats.dino([A, B, t, s])
            per.append({
                'stem': stem,
                'gt_true': float(f[2] @ f[1]),
                'ref_gain': float(f[2] @ f[1] - f[3] @ f[1]),
                'copy_gap': float(f[2] @ f[0] - f[1] @ f[0]),
                'copy': int(dhash_ham(t, A) <= args.copy_dhash),
                'ccip_true': feats.ccip_same(t, B),
            })
            rows.append([A, B, t, s])
        if not per:
            print(f'{d}: no complete rows')
            continue

        def mean(k):
            v = [p[k] for p in per if p[k] is not None]
            return round(sum(v) / len(v), 4) if v else None
        summary = {'dir': str(d), 'n': len(per), **{k: mean(k) for k in ('gt_true', 'ref_gain', 'copy_gap', 'ccip_true')},
                   'copy_rate': mean('copy')}
        (d / 'metrics.json').write_text(json.dumps({'summary': summary, 'pairs': per}, indent=2))
        g = labeled_grid(rows, [row['stem'] for row in per], f'{d.parent.name} | {d.name}')
        g.save(d / 'grid.png')
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
