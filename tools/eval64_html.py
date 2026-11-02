#!/usr/bin/env python3
"""Gera /workspace/eval64.html: referência, alvo e as gerações de cada run no eval64 (2 seeds), com métricas.

  python tools/eval64_html.py   # recria com todas as runs que já existem
"""
import html
import json
import os
from pathlib import Path

OUT = Path('/workspace/eval64.html')
CASES = '/workspace/runs/beta1r_samples_inputs/cases_eval64.json'
RUNS = [  # (rótulo, pasta de samples do eval64)
    ('BETA1-R 750', '/workspace/runs/beta1r_samples/eval64_baseline/step750'),
    ('BETA1-R 1000', '/workspace/runs/beta1r_samples/eval64_baseline/step1000'),
]
for step in (250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000):
    RUNS.append((f'DELTA1000 {step}', f'/workspace/runs/delta1000_samples/step{step}/eval64'))
RUNS += [
]
for step in (500, 1000, 1500, 2000):
    RUNS.append((f'R4a {step}', f'/workspace/runs/beta1r4a_samples/step{step}/eval64'))
    RUNS.append((f'R4b {step}', f'/workspace/runs/beta1r4b_samples/step{step}/eval64'))


def rel(p):
    return html.escape(os.path.relpath(p, OUT.parent))


def main():
    cases = json.load(open(CASES))
    runs = []
    for label, d in RUNS:
        m = Path(d) / 'metrics_eval.json'
        if m.exists():
            met = json.load(open(m))
            rows = {(r['name'], r['seed']): r for r in met['rows']}
            runs.append((label, Path(d), met, rows))
    parts = ['<!doctype html><meta charset="utf-8"><title>eval64</title><style>',
             'body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#ddd}',
             'table{border-collapse:collapse}td,th{padding:4px;vertical-align:top;border-bottom:1px solid #333}',
             'img{height:150px;display:block;border:3px solid transparent}img.copy{border-color:#e33}',
             '.m{font-size:11px;color:#aaa}.c{color:#f55;font-weight:bold}.p{font-size:11px;max-width:260px;color:#bbb}',
             'th{position:sticky;top:0;background:#222}.sum{font-size:13px;margin:4px 0}</style>',
             '<h2>eval64: 64 casos fora do treino, 2 seeds (76 e 1)</h2>',
             '<p class="m">Borda vermelha = cópia (dHash ≤ 10 ou DINO ≥ 0,95 contra a referência). '
             'Abaixo de cada imagem: dHash / DINO contra a referência | DINO contra o alvo.</p>']
    for label, _, met, _ in runs:
        parts.append(f'<div class="sum"><b>{html.escape(label)}</b>: {html.escape(met["summary"])}</div>')
    parts.append('<table><tr><th>caso</th><th>referência</th><th>alvo real</th>')
    for label, *_ in runs:
        parts.append(f'<th colspan="2">{html.escape(label)}</th>')
    parts.append('</tr>')
    order = sorted(cases, key=lambda c: (c.get('stratum', ''), c['name']))
    for c in order:
        parts.append(f'<tr><td><b>{html.escape(c["name"])}</b><br><span class="m">{c.get("stratum", "")}</span>'
                     f'<div class="p">{html.escape(c["prompt"])}</div></td>')
        parts.append(f'<td><img src="{rel(c["reference"])}"></td><td><img src="{rel(c["target"])}"></td>')
        for label, d, met, rows in runs:
            for seed, path in ((76, d / f'{c["name"]}.png'), (1, d / 'seeds' / f'{c["name"]}_s1.jpg')):
                r = rows.get((c['name'], seed))
                if not path.exists() or r is None:
                    parts.append('<td></td>')
                    continue
                cls = ' class="copy"' if r['copy'] else ''
                tag = '<span class="c">CÓPIA</span> ' if r['copy'] else ''
                parts.append(f'<td><img{cls} src="{rel(path)}"><div class="m">{tag}s{seed}: {r["dhash_ref"]} / '
                             f'{r["dino_ref"]:.2f} | alvo {r["dino_tgt"]:.2f}</div></td>')
        parts.append('</tr>')
    parts.append('</table>')
    OUT.write_text('\n'.join(parts))
    print(f'{OUT} ({len(runs)} runs: {", ".join(r[0] for r in runs)})')


if __name__ == '__main__':
    main()
