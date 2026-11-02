#!/usr/bin/env python3
"""Calibra o limiar da máscara espacial anti-cópia (BETA1-R4) no espaço latente.

Verdade de referência (revisão independente, overlap.py): um bloco do alvo é "cópia alinhada" quando a média de
|A-B| (RGB em [0,1], referência com recorte central + resize para o tamanho do alvo, como no trainer) no bloco
8x8 px que vira 1 pixel latente é < 0.08. No treino só há latentes, então procura-se um limiar em
d_lat = média_c |z_alvo - z_ref| (latentes normalizados Wan21.process_in, como no cache) que reproduza esse mapa.

Saída: tabela de F1 por limiar e o melhor limiar (JSON em --out).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils.common  # noqa: F401,E402
import models.base  # noqa: F401,E402  (põe o ComfyUI no sys.path)
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.latent_formats  # noqa: E402

DATA = Path('/workspace/data/beta1r')
VAE = '/dev/shm/models/krea2/vae/qwen_image_vae.safetensors'
BUCKETS = [(512, 512), (576, 448), (640, 400), (720, 368), (448, 576), (400, 640), (368, 720)]


def fit(im, w, h):
    W, H = im.size
    ar = w / h
    if W / H > ar:
        nw = int(H * ar)
        x0 = (W - nw) // 2
        im = im.crop((x0, 0, x0 + nw, H))
    else:
        nh = int(W / ar)
        y0 = (H - nh) // 2
        im = im.crop((0, y0, W, y0 + nh))
    return im.resize((w, h), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    random.seed(0)
    jobs = []
    srcs = sorted(p.name for p in DATA.iterdir() if (p / 'target').is_dir())
    for src in srcs:
        fs = [f for f in os.listdir(DATA / src / 'target') if f.endswith('.jpg')]
        random.shuffle(fs)
        for f in fs[: a.n // len(srcs) + 1]:
            r = DATA / src / 'refs' / f'{f[:-4]}_1.jpg'
            if r.exists():
                jobs.append((src, DATA / src / 'target' / f, r))
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
    px_maps, lat_maps, srcs_of = [], [], []
    for src, t, r in jobs:
        T = Image.open(t).convert('RGB')
        ar = T.width / T.height
        w, h = min(BUCKETS, key=lambda b: abs(np.log(b[0] / b[1]) - np.log(ar)))
        tt, rr = fit(T, w, h), fit(Image.open(r).convert('RGB'), w, h)
        A = np.asarray(tt, dtype=np.float32) / 255
        B = np.asarray(rr, dtype=np.float32) / 255
        d = np.abs(A - B).mean(-1)
        px = d.reshape(h // 8, 8, w // 8, 8).mean((1, 3))
        with torch.no_grad():
            # VAE de vídeo: uma imagem por chamada (um batch de 2 viraria 2 frames)
            za, zb = (vae.encode(torch.from_numpy(x[None])).float() for x in (A, B))
        # mesmo espaço do cache do trainer (models/base.py vae_encode: Wan21.process_in)
        za, zb = (comfy.latent_formats.Wan21().process_in(z) for z in (za, zb))
        za, zb = (z.reshape(z.shape[1], -1, *z.shape[-2:])[:, 0] for z in (za, zb))
        lat = (za - zb).abs().mean(0).cpu().numpy()
        if lat.shape != px.shape:
            continue
        px_maps.append(px)
        lat_maps.append(lat)
        srcs_of.append(src)
    gt = np.concatenate([(p < 0.08).ravel() for p in px_maps])
    lat_all = np.concatenate([l.ravel() for l in lat_maps])
    rows = []
    for thr in np.quantile(lat_all, np.linspace(0.05, 0.7, 40)):
        pred = lat_all < thr
        tp = (pred & gt).sum()
        prec = tp / max(1, pred.sum())
        rec = tp / max(1, gt.sum())
        rows.append({'thr': float(thr), 'f1': float(2 * prec * rec / max(1e-9, prec + rec)),
                     'precision': float(prec), 'recall': float(rec), 'frac_masked': float(pred.mean())})
    best = max(rows, key=lambda x: x['f1'])
    per_pair_gt = [float((p < 0.08).mean()) for p in px_maps]
    per_pair_pred = [float((l < best['thr']).mean()) for l in lat_maps]
    out = {'n_pairs': len(px_maps), 'gt_aligned_frac_mean': float(np.mean(per_pair_gt)),
           'pred_aligned_frac_mean': float(np.mean(per_pair_pred)),
           'pair_corr': float(np.corrcoef(per_pair_gt, per_pair_pred)[0, 1]),
           'best': best, 'table': rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'table'}, indent=1))


if __name__ == '__main__':
    main()
