#!/usr/bin/env python3
"""Próxima cena, dataset grande delta — etapa 1: varredura de pares (filtro de cópia + hashes).

Para cada par (BETA1-R e v11): área alinhada quase idêntica entre alvo e referência (ref com recorte central para
o aspecto do alvo, 64 px no lado maior, média RGB |A-B| < 0.08) e dHash 8x8 de alvo e referência (para o filtro
de quase-idênticos e para casar os pares do v11 com os do BETA1-R).

  python tools/delta_big_scan.py /workspace/data/delta_big/scan.jsonl
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

B1R = '/workspace/data/beta1r'
V11 = '/workspace/data/v11/natural'


def fit(im, w, h):
    W, H = im.size
    ar = w / h
    if W / H > ar:
        nw = int(H * ar)
        im = im.crop(((W - nw) // 2, 0, (W - nw) // 2 + nw, H))
    else:
        nh = int(W / ar)
        im = im.crop((0, (H - nh) // 2, W, (H - nh) // 2 + nh))
    return np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255


def dhash(im):
    g = np.asarray(im.convert('L').resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return int(''.join('1' if b else '0' for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def work(job):
    src, stem, tgt, ref = job
    try:
        T = Image.open(tgt).convert('RGB')
        R = Image.open(ref).convert('RGB')
        s = 64 / max(T.size)
        w, h = max(8, int(T.width * s)), max(8, int(T.height * s))
        al = float((np.abs(fit(T, w, h) - fit(R, w, h)).mean(-1) < 0.08).mean())
        return {'src': src, 'stem': stem, 'target': tgt, 'ref': ref, 'aligned': al,
                'dh_t': dhash(T), 'dh_r': dhash(R), 'dh_tr': bin(dhash(T) ^ dhash(R)).count('1')}
    except Exception as e:  # noqa
        return {'src': src, 'stem': stem, 'target': tgt, 'ref': ref, 'error': str(e)}


def jobs():
    out = []
    for src in sorted(os.listdir(B1R)):
        td, rd = f'{B1R}/{src}/target', f'{B1R}/{src}/refs'
        if not os.path.isdir(td):
            continue
        for fn in sorted(os.listdir(td)):
            if not fn.endswith('.jpg'):
                continue
            stem = fn[:-4]
            ref = f'{rd}/{stem}_1.jpg'
            if os.path.exists(ref):
                out.append((src, stem, f'{td}/{fn}', ref))
    for fn in sorted(os.listdir(f'{V11}/images_B')):
        a = f'{V11}/images_A/{fn}'
        if os.path.exists(a):
            out.append(('v11', fn.rsplit('.', 1)[0], f'{V11}/images_B/{fn}', a))
    return out


if __name__ == '__main__':
    J = jobs()
    with Pool(32) as p:
        res = p.map(work, J, chunksize=32)
    with open(sys.argv[1], 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    print(len(J), 'pares', sum('error' in r for r in res), 'erros')
