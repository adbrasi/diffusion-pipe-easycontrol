#!/usr/bin/env python3
"""Sensibilidade por SEED — testa se as conclusões da bateria sobrevivem a
variação de ruído inicial.

A bateria inteira foi julgada com uma seed só (76). Se o ranking entre
braços muda ao trocar a seed, as diferenças medidas eram ruído.
"""
import sys, os, glob, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_metrics import _load, dist

ARMS = {
    'arm1 s500':  ('/workspace/outputs/arm1_broad_llm_frozen', 's500'),
    'armD s2000': ('/workspace/outputs/armD_dropout_2000', 's2000'),
}
SEEDS = ['', '_seed1234', '_seed777', '_seed42', '_seed2024', '_seed555']

print(f'{"braço":<13}{"seed":>9}{"ex1":>8}{"ex2":>8}{"ex3":>8}{"média":>9}')
print('-' * 56)
summary = {}
for name, (d, step) in ARMS.items():
    per_seed = []
    for sfx in SEEDS:
        vals = []
        for ex in ('ex1', 'ex2', 'ex3'):
            a = f'{d}/{step}_{ex}_ref1.0{sfx}.png'
            b = f'{d}/{step}_{ex}_refshuffle{sfx}.png'
            if os.path.exists(a) and os.path.exists(b):
                vals.append(dist(_load(a), _load(b)))
        if len(vals) == 3:
            m = sum(vals) / 3
            per_seed.append(m)
            lbl = sfx.replace('_seed', '') or '76'
            print(f'{name:<13}{lbl:>9}' + ''.join(f'{v:>8.3f}' for v in vals) + f'{m:>9.3f}')
    summary[name] = per_seed

print()
for name, ps in summary.items():
    if len(ps) >= 2:
        sd = statistics.stdev(ps)
        print(f'{name:<13} média entre seeds = {sum(ps)/len(ps):.3f}  desvio = {sd:.3f}  '
              f'(min {min(ps):.3f} / max {max(ps):.3f})')
if all(len(v) >= 2 for v in summary.values()):
    a, b = summary['arm1 s500'], summary['armD s2000']
    gap = (sum(b)/len(b)) - (sum(a)/len(a))
    pooled = statistics.stdev(a + b)
    print(f'\ndiferença armD-arm1 = {gap:+.3f} | desvio combinado entre seeds = {pooled:.3f}')
    # Welch t-test (nao assume variancias iguais — o armD varia bem mais)
    na, nb = len(a), len(b)
    if na >= 2 and nb >= 2:
        va, vb = statistics.variance(a), statistics.variance(b)
        se = (va/na + vb/nb) ** 0.5
        t = gap / se if se else float('inf')
        # graus de liberdade de Welch-Satterthwaite
        df = ((va/na + vb/nb) ** 2) / (
            (va/na) ** 2 / (na-1) + (vb/nb) ** 2 / (nb-1)) if se else 1
        print(f'Welch t = {t:.2f}, df ~ {df:.1f}, n = {na} vs {nb} seeds')
        # t critico bilateral a 5% para df pequeno (tabela abreviada)
        crit = {1:12.71, 2:4.30, 3:3.18, 4:2.78, 5:2.57, 6:2.45, 7:2.36,
                8:2.31, 9:2.26, 10:2.23}.get(max(1, min(10, round(df))), 2.23)
        print(f't critico (5%, bilateral) ~ {crit}')
        print('CONCLUSIVO: armD > arm1' if t > crit else
              'NAO CONCLUSIVO — a diferenca nao se separa do ruido entre seeds')
