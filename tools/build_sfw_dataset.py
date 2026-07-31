#!/usr/bin/env python3
"""Monta a arvore do dataset 2 (SFW) para o treino do zero em 1024px.

Contexto: o adapter da fase 1 ficou enviesado porque o dataset de 30k e
dominado por NSFW. O dataset 2 e uma selecao SFW de ~11.000 pares definida
pelo usuario:

  poxima_cena_v2  ~6.000  todos
  recortados       2.500  amostra aleatoria
  comikontext      2.500  amostra aleatoria, EXCLUINDO tudo abaixo do par
                          000800 (regra que vale so para esta fonte)

Por que hardlink e nao copia: as fontes ja estao normalizadas em
/workspace/datasets/prepared/ desde a fase 1, no mesmo filesystem. Hardlink
custa zero de disco e mantem as imagens vivas mesmo depois que os caches (e
ate as pastas) da fase 1 forem apagados.

Por que uma arvore nova e nao reaproveitar prepared/: o treino da fase 1 le
prepared/ e o supervisor relanca o trainer a cada 500 steps; mexer naquelas
pastas faria o trainer cachear pares novos no meio do run.

Layout produzido:
  <out>/poxima/{target,refs}
  <out>/recortados/{target,refs}
  <out>/comik/{target,refs}

Cada target/ tem <stem>.jpg + <stem>.txt; cada refs/ tem <stem>_1.jpg.
O cache de latentes/text-embeddings e criado dentro de cada target/cache
pelo proprio trainer, sem colidir com o da fase 1.

Uso:
  python tools/build_sfw_dataset.py --out /workspace/datasets/sfw [--dry-run]
"""
import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

PREPARED = Path('/workspace/datasets/prepared')

# (nome no dataset 2, pasta em prepared/, quantos pares, indice minimo)
# indice_min = None  -> sem filtro de indice
FONTES = [
    ('recortados', 'recortados', 2500, None),
    ('comik', 'mega3', 2500, 800),
]
SEED = 42


def stems_de(src: Path):
    """Stems com target .jpg + .txt + ref _1.jpg presentes (par completo)."""
    target, refs = src / 'target', src / 'refs'
    out = []
    for f in os.listdir(target):
        if not f.endswith('.jpg'):
            continue
        stem = f[:-4]
        if (target / f'{stem}.txt').exists() and (refs / f'{stem}_1.jpg').exists():
            out.append(stem)
    return sorted(out)


def indice(stem: str):
    """Numero do par no nome {prefix}_{stem_original}.

    prepare_krea2_edit.py preserva o stem original, entao comik_000800 e
    literalmente o par 000800 do comikontext.
    """
    m = re.search(r'_(\d+)$', stem)
    return int(m.group(1)) if m else None


def ligar(src_dir: Path, dst_dir: Path, stems, dry=False):
    """Hardlink dos 3 arquivos de cada par. Cai para copia entre filesystems."""
    (dst_dir / 'target').mkdir(parents=True, exist_ok=True)
    (dst_dir / 'refs').mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in stems:
        pares = [
            (src_dir / 'target' / f'{stem}.jpg', dst_dir / 'target' / f'{stem}.jpg'),
            (src_dir / 'target' / f'{stem}.txt', dst_dir / 'target' / f'{stem}.txt'),
            (src_dir / 'refs' / f'{stem}_1.jpg', dst_dir / 'refs' / f'{stem}_1.jpg'),
        ]
        if dry:
            n += 1
            continue
        for s, d in pares:
            if d.exists():
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/workspace/datasets/sfw')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    rng = random.Random(SEED)

    total = 0
    for nome, pasta, quantos, idx_min in FONTES:
        src = PREPARED / pasta
        if not src.exists():
            print(f'ERRO: {src} nao existe', file=sys.stderr)
            sys.exit(1)
        stems = stems_de(src)
        antes = len(stems)
        if idx_min is not None:
            stems = [s for s in stems if (indice(s) or 0) >= idx_min]
        if len(stems) < quantos:
            print(f'ERRO: {nome} tem {len(stems)} pares elegiveis, '
                  f'precisa de {quantos}', file=sys.stderr)
            sys.exit(1)
        escolhidos = sorted(rng.sample(stems, quantos))
        n = ligar(src, out / nome, escolhidos, dry=args.dry_run)
        filtro = f' (>= {idx_min}: {len(stems)} de {antes})' if idx_min else f' (de {antes})'
        print(f'{nome}: {n} pares{filtro} -> {out / nome}')
        # rastro do que foi escolhido, para reproduzir/auditar depois
        if not args.dry_run:
            (out / nome / 'selecao.txt').write_text('\n'.join(escolhidos) + '\n')
        total += n

    print(f'\ntotal ligado desta etapa: {total}')
    print('falta o poxima: preparar de /workspace/datasets/raw_v2/poxima_cena_v2 '
          f'com prepare_krea2_edit.py ab_dirs -> {out}/poxima')


if __name__ == '__main__':
    main()
