#!/usr/bin/env python3
"""Monta o JSON de sampling do dataset 2 (SFW).

CLAUDE.md §7.3: os prompts de sampling saem de itens que JA estao no dataset,
com a caption usada como esta. Nada e inventado e nada e filtrado por
conteudo — a escolha e mecanica (1 par por fonte, seed fixa) para ser
reproduzivel e auditavel.

O tamanho de cada sample vem do aspect ratio da imagem ALVO, reescalado para
area ~1024^2 e arredondado para multiplo de 32 (o VAE reduz 8x e o patch e 2).

Uso:
  python tools/make_sfw_samples.py --out /workspace/outputs/sampling_inputs/samples_sfw.json
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

RAIZ = Path('/workspace/datasets/sfw')
FONTES = ['poxima', 'recortados', 'comik']
AREA = 1024 * 1024
SEED = 42


def dimensoes(img_path: Path):
    with Image.open(img_path) as im:
        w, h = im.size
    ar = w / h
    nh = (AREA / ar) ** 0.5
    nw = nh * ar
    return max(256, int(round(nw / 32)) * 32), max(256, int(round(nh / 32)) * 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/workspace/outputs/sampling_inputs/samples_sfw.json')
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    samples = []
    for i, fonte in enumerate(FONTES):
        base = RAIZ / fonte
        stems = sorted(p.stem for p in (base / 'target').glob('*.jpg'))
        if not stems:
            raise SystemExit(f'{fonte}: nenhum par em {base}')
        stem = rng.choice(stems)
        alvo = base / 'target' / f'{stem}.jpg'
        ref = base / 'refs' / f'{stem}_1.jpg'
        caption = (base / 'target' / f'{stem}.txt').read_text().strip()

        # copia estavel: o dataset pode ser reorganizado, o input do sampling nao
        ref_dst = out.parent / f'sfw_example{i}_ref.jpg'
        alvo_dst = out.parent / f'sfw_example{i}_target.jpg'
        shutil.copy2(ref, ref_dst)
        shutil.copy2(alvo, alvo_dst)

        w, h = dimensoes(alvo)
        samples.append({
            'source': fonte,
            'stem': stem,
            'reference': str(ref_dst),
            'target_original': str(alvo_dst),
            'prompt': caption,
            'width': w,
            'height': h,
        })
        print(f'{fonte}: {stem} -> {w}x{h}')

    out.write_text(json.dumps({
        'seed_picker': SEED,
        'nota': 'dataset 2 (SFW); 1 par por fonte, escolha mecanica, caption como esta',
        'samples': samples,
    }, indent=1, ensure_ascii=False))
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
