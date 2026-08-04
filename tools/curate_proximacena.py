#!/usr/bin/env python3
"""Curadoria dos datasets do run k2_proximacena_v2.

Entrada: pastas com images_A (control) + images_B (target), pareadas por nome.
Saída: /workspace/datasets/proxima_cena/<nome>/{images_A,images_B} com
hardlinks (custo zero de disco), stems prefixados e SEM os .txt antigos.

Uso:
  python curate_proximacena.py SRC DEST PREFIX [--sample N] [--seed 42]
"""
import argparse
import os
import random
import sys
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path, help='pasta contendo images_A e images_B')
    ap.add_argument('dest', type=Path)
    ap.add_argument('prefix')
    ap.add_argument('--sample', type=int, default=0, help='N pares aleatórios (0 = todos)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    a_dir, b_dir = args.src / 'images_A', args.src / 'images_B'
    a = {p.stem: p for p in a_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    b = {p.stem: p for p in b_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    stems = sorted(set(a) & set(b))
    orphans = len(set(a) ^ set(b))
    print(f'{args.src}: {len(stems)} pares casados, {orphans} órfãos ignorados')

    if args.sample and len(stems) > args.sample:
        rng = random.Random(args.seed)
        stems = sorted(rng.sample(stems, args.sample))
        print(f'amostrados {len(stems)} pares (seed {args.seed})')

    out_a = args.dest / 'images_A'
    out_b = args.dest / 'images_B'
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    def sanitize(stem):
        return ''.join(c if c.isalnum() or c in '-_' else '-' for c in stem)

    n = 0
    for stem in stems:
        new = f'{args.prefix}_{sanitize(stem)}'
        for src, out in ((a[stem], out_a), (b[stem], out_b)):
            dst = out / (new + src.suffix.lower())
            if not dst.exists():
                os.link(src, dst)
        n += 1
    print(f'{n} pares em {args.dest} (prefixo {args.prefix})')
    sys.exit(0)


if __name__ == '__main__':
    main()
