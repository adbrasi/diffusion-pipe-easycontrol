#!/usr/bin/env python3
"""Métricas de avaliação anti-cópia (revisão independente, 02/11/2026) para uma pasta de samples do
tools/beta1r_sample.py (seed principal em <dir>/<name>.png, seeds extras em <dir>/seeds/<name>_s<seed>.jpg).

Por geração: dHash(gen, ref), dHash(gen, alvo), DINOv2 cos(gen, ref), DINOv2 cos(gen, alvo) e distância de paleta
(histogramas Lab, L1/2 em [0, 1]) contra a referência.
A unidade estatística é o CASO (os seeds de um caso são muito correlacionados): um caso "copia" se a maioria dos
seeds copia (e também se qualquer seed copia). Cópia por dHash (<= 10) e por DINO (cos(gen, ref) >= 0.95), em
separado e combinadas (OU). IC 95 % por bootstrap sobre casos, total e por estrato (cases_eval64.json: "stratum").

  python tools/beta1r_eval_metrics.py --samples /workspace/runs/beta1r_samples/step750 \
      --cases /workspace/runs/beta1r_samples_inputs/cases.json
Saída: <samples>/metrics_eval.json + resumo de uma linha no stdout. Importável: evaluate(samples, cases).
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

DHASH_THR = 10
DINO_THR = 0.95
DINO_MODEL = 'facebook/dinov2-small'
_DINO = {}


def dhash_bits(img, size=8):
    g = np.asarray(img.convert('L').resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def palette_hist(img, bins=32):
    lab = np.asarray(img.convert('RGB').resize((128, 128)).convert('LAB'), dtype=np.float32)
    hs = [np.histogram(lab[..., c], bins=bins, range=(0, 255))[0].astype(np.float32) for c in range(3)]
    return [h / max(1.0, h.sum()) for h in hs]


def palette_dist(ha, hb):
    return float(np.mean([0.5 * np.abs(a - b).sum() for a, b in zip(ha, hb)]))


def _dino(device):
    if device not in _DINO:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        os.environ.setdefault('HF_HOME', '/workspace/.hf_home')
        proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
        model = AutoModel.from_pretrained(DINO_MODEL).eval().to(device)
        _DINO[device] = (proc, model)
    return _DINO[device]


def dino_embed(images, device):
    import torch
    proc, model = _dino(device)
    out = []
    for i in range(0, len(images), 16):
        with torch.no_grad():
            batch = proc(images=[im.convert('RGB') for im in images[i:i + 16]], return_tensors='pt').to(device)
            out.append(torch.nn.functional.normalize(model(**batch).pooler_output.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def pick_device(device):
    if device != 'auto':
        return device
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4 * 2 ** 30:
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def bootstrap(values, n=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'ci95': [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            'n': int(len(values))}


def collect(samples, cases, all_cases=False):
    samples = Path(samples)
    main_seed = 76
    mpath = samples / 'metrics.json'
    if mpath.exists():
        main_seed = json.load(open(mpath)).get('seed', 76)
    gens = []
    for c in cases:
        if not all_cases and not c.get('copy_eval', False):
            continue
        p = samples / f"{c['name']}.png"
        if p.exists():
            gens.append((c, main_seed, p))
        for q in sorted((samples / 'seeds').glob(f"{c['name']}_s*.jpg")) if (samples / 'seeds').is_dir() else []:
            seed = q.stem[len(c['name']) + 2:]
            if seed.isdigit():
                gens.append((c, int(seed), q))
    return gens


def evaluate(samples, cases_path, device='auto', all_cases=False, write=True):
    cases = json.load(open(cases_path))
    gens = collect(samples, cases, all_cases)
    if not gens:
        raise RuntimeError(f'nenhuma geração de {cases_path} em {samples}')
    device = pick_device(device)
    names = sorted({c['name'] for c, _, _ in gens})
    by_name = {c['name']: c for c in cases}
    refs = {n: Image.open(by_name[n]['reference']).convert('RGB') for n in names}
    tgts = {n: Image.open(by_name[n]['target']).convert('RGB') for n in names if by_name[n].get('target')}
    gen_imgs = [Image.open(p).convert('RGB') for _, _, p in gens]
    e_gen = dino_embed(gen_imgs, device)
    e_ref = dict(zip(names, dino_embed([refs[n] for n in names], device)))
    tnames = [n for n in names if n in tgts]
    e_tgt = dict(zip(tnames, dino_embed([tgts[n] for n in tnames], device))) if tnames else {}
    ref_bits = {n: dhash_bits(refs[n]) for n in names}
    tgt_bits = {n: dhash_bits(tgts[n]) for n in tnames}
    ref_pal = {n: palette_hist(refs[n]) for n in names}
    rows = []
    for (c, seed, p), img, e in zip(gens, gen_imgs, e_gen):
        n = c['name']
        b = dhash_bits(img)
        row = {'name': n, 'seed': seed, 'stratum': c.get('stratum', 'all'),
               'dhash_ref': int((b != ref_bits[n]).sum()),
               'dino_ref': float(e @ e_ref[n]),
               'palette_ref': palette_dist(palette_hist(img), ref_pal[n])}
        if n in tgt_bits:
            row['dhash_tgt'] = int((b != tgt_bits[n]).sum())
            row['dino_tgt'] = float(e @ e_tgt[n])
        row['copy_dhash'] = row['dhash_ref'] <= DHASH_THR
        row['copy_dino'] = row['dino_ref'] >= DINO_THR
        row['copy'] = row['copy_dhash'] or row['copy_dino']
        rows.append(row)
    per_case = []
    for n in names:
        rs = [r for r in rows if r['name'] == n]
        pc = {'name': n, 'stratum': rs[0]['stratum'], 'n_seeds': len(rs)}
        for k in ('dhash_ref', 'dino_ref', 'palette_ref', 'dhash_tgt', 'dino_tgt'):
            vals = [r[k] for r in rs if k in r]
            if vals:
                pc[k] = float(np.mean(vals))
        for k in ('copy_dhash', 'copy_dino', 'copy'):
            frac = float(np.mean([r[k] for r in rs]))
            pc[f'{k}_frac'] = frac
            pc[f'{k}_majority'] = frac > 0.5
            pc[f'{k}_any'] = frac > 0
        per_case.append(pc)

    def summarize(pcs):
        out = {'n_cases': len(pcs)}
        for k in ('copy_majority', 'copy_any', 'copy_dhash_majority', 'copy_dino_majority'):
            out[k] = bootstrap([float(p[k]) for p in pcs])
        for k in ('dhash_ref', 'dino_ref', 'palette_ref', 'dhash_tgt', 'dino_tgt'):
            vals = [p[k] for p in pcs if k in p]
            if vals:
                out[k] = bootstrap(vals)
        out['copied_cases'] = sorted(p['name'] for p in pcs if p['copy_majority'])
        return out

    strata = sorted({p['stratum'] for p in per_case})
    result = {'samples': str(samples), 'cases': str(cases_path), 'device': device,
              'thresholds': {'dhash_ref<=': DHASH_THR, 'dino_ref>=': DINO_THR},
              'n_generations': len(rows), 'all': summarize(per_case),
              'by_stratum': {s: summarize([p for p in per_case if p['stratum'] == s]) for s in strata},
              'per_case': per_case, 'rows': rows}
    result['summary'] = summary_line(result)
    if write:
        json.dump(result, open(Path(samples) / 'metrics_eval.json', 'w'), indent=1)
    return result


def summary_line(r):
    a = r['all']

    def fmt(s):
        return f"{s['mean']:.2f}[{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]" if s else '-'
    parts = [f"casos={a['n_cases']} gens={r['n_generations']}",
             f"cópia(caso,maioria)={fmt(a['copy_majority'])} qualquer_seed={fmt(a['copy_any'])}",
             f"dino_ref={fmt(a.get('dino_ref'))} dino_alvo={fmt(a.get('dino_tgt'))} dhash_ref={fmt(a.get('dhash_ref'))}"]
    for s, v in r['by_stratum'].items():
        if s != 'all':
            parts.append(f"{s}: cópia={fmt(v['copy_majority'])} dino_ref={fmt(v.get('dino_ref'))}")
    return ' | '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--all-cases', action='store_true', help='inclui casos sem copy_eval (ex.: exemplos do beta1)')
    a = ap.parse_args()
    random.seed(0)
    r = evaluate(a.samples, a.cases, a.device, a.all_cases)
    print(r['summary'])


if __name__ == '__main__':
    main()
