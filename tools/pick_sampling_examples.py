#!/usr/bin/env python3
"""Escolhe 3 exemplos ALEATÓRIOS do dataset preparado para o sampling do treino.

Regras (CLAUDE.md §7.3): prompts vêm das captions do próprio dataset, usadas
como estão. Registra o input exato (ref + prompt) em samples.json e copia as
imagens para o diretório de sampling_inputs (que sobe para o HF).

Uso: pick_sampling_examples.py <prepared_root> <out_dir> [--seed 42] [--n 3]
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prepared_root', type=Path)
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n', type=int, default=3)
    args = ap.parse_args()

    pairs = []
    for target_dir in sorted(args.prepared_root.glob('*/target')):
        refs_dir = target_dir.parent / 'refs'
        for txt in target_dir.glob('*.txt'):
            img = txt.with_suffix('.jpg')
            ref = refs_dir / f'{txt.stem}_1.jpg'
            if img.exists() and ref.exists():
                pairs.append((img, ref, txt))
    print(f'{len(pairs)} pares candidatos')

    rng = random.Random(args.seed)
    picked = rng.sample(pairs, args.n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, (img, ref, txt) in enumerate(picked):
        ref_copy = args.out_dir / f'example{i}_ref.jpg'
        shutil.copy2(ref, ref_copy)
        shutil.copy2(img, args.out_dir / f'example{i}_target_groundtruth.jpg')
        prompt = txt.read_text(encoding='utf-8').strip()
        (args.out_dir / f'example{i}_prompt.txt').write_text(prompt, encoding='utf-8')
        with Image.open(img) as im:
            w, h = im.size
        # tamanho de geração: AR do target, área ~512², múltiplos de 16
        scale = (512 * 512 / (w * h)) ** 0.5
        gw = max(256, round(w * scale / 16) * 16)
        gh = max(256, round(h * scale / 16) * 16)
        samples.append({'source': str(img.parent.parent.name), 'stem': txt.stem,
                        'reference': str(ref_copy), 'prompt': prompt,
                        'width': gw, 'height': gh})
        print(f'example{i}: {txt.stem} ({samples[-1]["source"]}) {gw}x{gh}')
    (args.out_dir / 'samples.json').write_text(
        json.dumps({'seed_picker': args.seed, 'samples': samples},
                   indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'-> {args.out_dir}/samples.json')


if __name__ == '__main__':
    main()
