#!/usr/bin/env python3
"""Próxima cena, dataset grande delta — etapa final: monta /workspace/data/delta_big/{target,refs} no formato do BETA1-R.

Junta selected.json (pares que passaram no filtro de cópia: área alinhada <= 0.30 e dHash alvo×ref > 4) com as legendas
(caption/reused.jsonl = delta do cap1000 reaproveitadas, caption/new.jsonl = Muse Spark 1.3 com prompt_final.txt);
o prefixo "Next scene of Picture 1." é removido de todas.
Normaliza como o BETA1-R: lado maior <= 1024, JPEG q95. Pares sem legenda válida ficam de fora.
"""
import json
import os
from multiprocessing import Pool

from PIL import Image

D = '/workspace/data/delta_big'
PREFIX = 'Next scene of Picture 1.'  # decisão do usuário: legendas finais sem o prefixo


def strip_prefix(c):
    c = c.strip()
    return c[len(PREFIX):].strip() if c.startswith(PREFIX) else c


def norm(src, dst):
    im = Image.open(src).convert('RGB')
    w, h = im.size
    s = min(1.0, 1024 / max(w, h))
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    im.save(dst, 'JPEG', quality=95)


def work(a):
    sel, cap = a
    stem = sel['key'].replace('/', '__')
    t, r = f'{D}/target/{stem}.jpg', f'{D}/refs/{stem}_1.jpg'
    if not os.path.exists(t):
        norm(sel['target'], t)
    if not os.path.exists(r):
        norm(sel['ref'], r)
    open(f'{D}/target/{stem}.txt', 'w').write(cap)
    return sel['src']


if __name__ == '__main__':
    os.makedirs(f'{D}/target', exist_ok=True)
    os.makedirs(f'{D}/refs', exist_ok=True)
    caps = {}
    for f in ('caption/reused.jsonl', 'caption/new.jsonl'):
        for l in open(f'{D}/{f}'):
            r = json.loads(l)
            if r.get('ok'):
                caps[r['key']] = strip_prefix(r['caption'])
    sel = [s for s in json.load(open(f'{D}/selected.json')) if s['key'] in caps]
    with Pool(24) as p:
        srcs = p.map(work, [(s, caps[s['key']]) for s in sel], chunksize=16)
    from collections import Counter
    print(len(sel), 'pares', Counter(srcs))
    with open(f'{D}/captions_final.jsonl', 'w') as f:
        for s in sel:
            f.write(json.dumps({'key': s['key'], 'src': s['src'], 'aligned': s['aligned'], 'caption': caps[s['key']]}) + '\n')
