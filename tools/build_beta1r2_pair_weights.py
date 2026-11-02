#!/usr/bin/env python3
"""Peso de loss por par (BETA1-R2) a partir da similaridade A-B (tools/beta1r2_pairsim.py).

Faixas (docs/BETA1R2.md):
  T0 idêntico   (mse64 < 5e-4 ou dHash <= 2)          -> 0.0
  T1 quase igual (dHash 3-5 ou mse64 < 2e-3)          -> 0.3
  T2 parecido   (dHash 6-10)                          -> 0.7
  resto (e pares sem medida)                          -> 1.0 (omitido do json)

Chave = nome do arquivo de referência ('<stem>_1.jpg'), único entre as fontes
(o stem já leva o prefixo da fonte). Os dados no disco não mudam.

  python tools/build_beta1r2_pair_weights.py data/beta1r2/pairsim.json data/beta1r2/pair_weights.json
"""
import collections
import json
import sys

W_T0, W_T1, W_T2 = 0.0, 0.3, 0.7


def tier_weight(dhash, mse):
    if dhash < 0:
        return 1.0
    if mse < 5e-4 or dhash <= 2:
        return W_T0
    if dhash <= 5 or mse < 2e-3:
        return W_T1
    if dhash <= 10:
        return W_T2
    return 1.0


def main(src, dst):
    rows = json.load(open(src))
    weights, counts, seen = {}, collections.Counter(), set()
    for source, target_name, dhash, mse in rows:
        key = target_name[:-4] + '_1.jpg'
        if key in seen:
            raise SystemExit(f'chave duplicada: {key}')
        seen.add(key)
        w = tier_weight(dhash, mse)
        counts[(source, w)] += 1
        if w != 1.0:
            weights[key] = w
    json.dump(dict(sorted(weights.items())), open(dst, 'w'), indent=0)
    total = collections.Counter()
    for (source, w), n in sorted(counts.items()):
        total[w] += n
    print(f'{len(rows)} pares; por peso: {dict(sorted(total.items()))}')
    for source in sorted({s for s, _ in counts}):
        print(f'  {source:12s} ' + ' '.join(f'{w}:{counts[(source, w)]}' for w in (W_T0, W_T1, W_T2, 1.0)))
    mean = sum(w * n for w, n in total.items()) / max(1, sum(total.values()))
    print(f'peso médio = {mean:.4f} -> {dst}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
