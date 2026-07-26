#!/usr/bin/env python3
"""Filtra os pares positivo/negativo do slider de timing por MEDIÇÃO.

POR QUE ISTO EXISTE: medido numa amostra de 12 pares, 3 não serviam —
`c_037` tinha delta 0 (o clipe já era fluido, então positivo ≈ negativo e não
há eixo), `c_024` tinha delta 6.4 (clipe quase estático, o mpdecimate não
tinha o que remover), e `c_030` tinha delta **NEGATIVO** (-27.7): o
"negativo" ficou com MAIS frames duplicados que o positivo, o que treinaria o
slider na direção CONTRÁRIA.

Um par com delta≈0 dilui o sinal; um com delta negativo o corrompe. A média
agregada (+30 pontos) escondia os dois casos.

MÉTRICA: fração de pares de frames consecutivos quase idênticos
(diff média < 0.0002 em luminância 160x90). É a assinatura de "frame
segurado" — o que caracteriza animação limitada.

Uso: filter_pairs.py [--min-pos 25] [--min-delta 20] [--apply]
Sem --apply só relata. Com --apply, move os pares reprovados para
`<dir>_rejeitados/`.
"""
import argparse
import glob
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

POS = '/workspace/datasets/animateka/videos'
NEG = '/workspace/datasets/animateka_neg/videos'


def dup_fraction(path, max_frames=48):
    """% de pares de frames consecutivos quase idênticos."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', 'scale=160:90',
             '-frames:v', str(max_frames), f'{d}/%03d.png'],
            capture_output=True,
        )
        files = sorted(glob.glob(f'{d}/*.png'))
        if len(files) < 8:
            return None
        frames = [np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255
                  for p in files]
    diff = np.array([float(np.abs(frames[i + 1] - frames[i]).mean())
                     for i in range(len(frames) - 1)])
    return float((diff < 0.0002).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-pos', type=float, default=25.0,
                    help='mínimo de frames segurados no POSITIVO (abaixo disso o '
                         'clipe já era fluido e não exemplifica o alvo)')
    ap.add_argument('--min-delta', type=float, default=20.0,
                    help='queda mínima de duplicatas pos->neg (o tamanho do eixo)')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    negs = sorted(glob.glob(f'{NEG}/*.mp4'))
    print(f'avaliando {len(negs)} pares…\n')
    aprovados, reprovados = [], []
    for i, neg in enumerate(negs, 1):
        base = os.path.basename(neg)
        pos = f'{POS}/{base}'
        if not os.path.exists(pos):
            reprovados.append((base, None, None, 'sem positivo'))
            continue
        dp, dn = dup_fraction(pos), dup_fraction(neg)
        if dp is None or dn is None:
            reprovados.append((base, dp, dn, 'ilegivel'))
            continue
        delta = dp - dn
        if dp < args.min_pos:
            reprovados.append((base, dp, dn, f'pos {dp:.0f}% < {args.min_pos:.0f}%'))
        elif delta < args.min_delta:
            motivo = 'delta INVERTIDO' if delta < 0 else f'delta {delta:.0f} < {args.min_delta:.0f}'
            reprovados.append((base, dp, dn, motivo))
        else:
            aprovados.append((base, dp, dn, delta))
        if i % 50 == 0:
            print(f'  {i}/{len(negs)} — aprovados {len(aprovados)}')

    print(f'\nAPROVADOS: {len(aprovados)}  |  REPROVADOS: {len(reprovados)}')
    if aprovados:
        d = np.array([a[3] for a in aprovados])
        print(f'  delta dos aprovados: media {d.mean():.1f} | min {d.min():.1f} | max {d.max():.1f}')
    invertidos = [r for r in reprovados if r[3] == 'delta INVERTIDO']
    if invertidos:
        print(f'  ATENCAO: {len(invertidos)} pares com delta INVERTIDO (treinariam ao contrario)')
    from collections import Counter
    print('  motivos:', dict(Counter(r[3].split(' ')[0] for r in reprovados)))

    if args.apply:
        for d in (POS, NEG):
            os.makedirs(f'{d}_rejeitados', exist_ok=True)
        for base, *_ in reprovados:
            for d in (POS, NEG):
                for ext in ('.mp4', '.txt'):
                    src = f'{d}/{base[:-4]}{ext}'
                    if os.path.exists(src):
                        shutil.move(src, f'{d}_rejeitados/{base[:-4]}{ext}')
        print(f'\n{len(reprovados)} pares movidos para *_rejeitados/')


if __name__ == '__main__':
    main()
