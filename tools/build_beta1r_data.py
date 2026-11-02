#!/usr/bin/env python3
"""Monta o dataset do BETA1-R (opção A, aprovada pelo usuário 2026-11-02).

beta1 (k2-context-rush-ofc-beta1) SEM pico-banana, mesmas contagens por fonte:
    recortados 6.937 · parents 4.815 · inscene 473      <- originais (legendas originais)
    poxima_cena_v2 4.129 · comikontext 4.950 · mega4 865 <- originais no MEGA sem chave;
        substitutos = mesmas fontes em AdwolfCzar/proxima_cena_grounded_original_dataset
        (ds2 / ds3 / ds4, re-legendados como descrição de B). ds3 só tem 2.900 -> usa todos.
Amostragem por fonte: random.Random(42).sample sobre os pares ordenados.
Normalização idêntica a tools/prepare_krea2_edit.py (lado longo <=1024 LANCZOS, RGB JPEG q95).
recortados/parents são lidos direto dos zips remotos do HF (range reads; sem zip em disco).

Uso: build_beta1r_data.py <dst> [--pcg /workspace/data/pcg] [--only fonte,...]
"""
import argparse
import io
import json
import os
import random
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_krea2_edit import IMG_EXTS, load_rgb, save_norm  # noqa: E402

COUNTS = {'recortados': 6937, 'parents': 4815, 'inscene': 473,
          'poxima_v2': 4129, 'comikontext': 4950, 'mega4': 865}
SEED = 42


def write_pair(dst, stem, tgt, ref, caption):
    caption = (caption or '').strip()
    if not caption:
        return 'caption vazia'
    t = load_rgb(io.BytesIO(tgt) if isinstance(tgt, bytes) else tgt)
    r = load_rgb(io.BytesIO(ref) if isinstance(ref, bytes) else ref)
    if t is None or r is None:
        return 'imagem ilegível'
    save_norm(t, dst / 'target' / f'{stem}.jpg')
    save_norm(r, dst / 'refs' / f'{stem}_1.jpg')
    (dst / 'target' / f'{stem}.txt').write_text(caption, encoding='utf-8')
    return None


def sample(pairs, n, name):
    pairs = sorted(pairs, key=lambda p: p[0])
    if n >= len(pairs):
        print(f'{name}: pedido {n}, disponível {len(pairs)} -> todos', flush=True)
        return pairs
    return random.Random(SEED).sample(pairs, n)


def run(name, dst, jobs, fn, workers=16):
    (dst / 'target').mkdir(parents=True, exist_ok=True)
    (dst / 'refs').mkdir(parents=True, exist_ok=True)
    rejects, done = [], 0
    with ThreadPoolExecutor(workers) as ex:
        for stem, err in ex.map(fn, jobs):
            done += 1
            if err:
                rejects.append((stem, err))
            if done % 500 == 0:
                print(f'  {name}: {done}/{len(jobs)}', flush=True)
    print(f'{name}: {len(jobs) - len(rejects)} ok, {len(rejects)} rejeitados', flush=True)
    if rejects:
        (dst / 'rejects.log').write_text(''.join(f'{s}\t{e}\n' for s, e in rejects))


def remote_zip_source(name, dst, repo_files, n):
    from huggingface_hub import HfFileSystem
    token = os.environ['HF_TOKEN']
    local = threading.local()

    def zf(path):
        cache = getattr(local, 'z', None)
        if cache is None:
            cache = local.z = {}
        if path not in cache:
            fs = HfFileSystem(token=token)
            cache[path] = zipfile.ZipFile(fs.open(path, 'rb', block_size=1 << 20))
        return cache[path]

    # find_pairs_ab: <stem>_B.<img> + <stem>_B.txt + <stem>_A.<img>; dedup por stem
    imgs, txts = {}, {}
    for path in repo_files:
        for member in zf(path).namelist():
            p = Path(member)
            if p.suffix.lower() in IMG_EXTS:
                imgs[p.stem] = (path, member)
            elif p.suffix.lower() == '.txt':
                txts[p.stem] = (path, member)
    pairs = []
    for stem_b, b in imgs.items():
        if not stem_b.endswith('_B'):
            continue
        base = stem_b[:-2]
        if base + '_A' in imgs and stem_b in txts:
            pairs.append((base, b, imgs[base + '_A'], txts[stem_b]))
    print(f'{name}: {len(pairs)} pares completos', flush=True)
    chosen = sample(pairs, n, name)

    def fn(job):
        base, b, a, t = job
        try:
            rd = lambda pm: zf(pm[0]).read(pm[1])
            cap = rd(t).decode('utf-8', errors='replace')
            return base, write_pair(dst, f'{name}_{base}', rd(b), rd(a), cap)
        except Exception as e:
            return base, f'erro: {e!r}'
    run(name, dst, chosen, fn)


def ab_dirs_source(name, dst, src, n):
    a_imgs = {p.stem: p for p in (src / 'images_A').iterdir() if p.suffix.lower() in IMG_EXTS}
    b_dir = src / 'images_B'
    caps = {}
    jl = src / 'captions.jsonl'
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get('caption') and not d.get('error'):
                    caps[Path(d['video']).stem] = d['caption'].strip()
    pairs = []
    for p in b_dir.iterdir():
        if p.suffix.lower() not in IMG_EXTS or p.stem not in a_imgs:
            continue
        txt = p.with_suffix('.txt')
        cap = txt.read_text(encoding='utf-8', errors='replace').strip() if txt.exists() else ''
        cap = cap or caps.get(p.stem, '')
        if cap:
            pairs.append((p.stem, p, a_imgs[p.stem], cap))
    print(f'{name}: {len(pairs)} pares completos em {src}', flush=True)
    chosen = sample(pairs, n, name)
    run(name, dst, chosen, lambda j: (j[0], write_pair(dst, f'{name}_{j[0]}', j[1], j[2], j[3])))


def inscene_source(name, dst, n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    jobs = []
    for split in ('train', 'validation'):
        pf = hf_hub_download('peteromallet/InScene-Dataset', f'data/{split}-00000-of-00001.parquet',
                             repo_type='dataset', token=os.environ.get('HF_TOKEN'))
        for row in pq.read_table(pf).to_pylist():
            g = lambda v: v['bytes'] if isinstance(v, dict) else v
            stem = f"inscene_{split}_{row.get('image_id') or ''}_{len(jobs):04d}".replace('/', '_')
            jobs.append((stem, g(row['target_image']), g(row['control_image']), row['prompt']))
    chosen = sample(jobs, n, name)
    run(name, dst, chosen, lambda j: (j[0], write_pair(dst, j[0], j[1], j[2], j[3])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dst')
    ap.add_argument('--pcg', default='/workspace/data/pcg')
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    root, pcg = Path(args.dst), Path(args.pcg)
    only = set(args.only.split(',')) if args.only else set(COUNTS)
    R = 'datasets/AdwolfCzar/'
    for name in COUNTS:
        if name not in only:
            continue
        dst = root / name
        if name == 'recortados':
            remote_zip_source(name, dst, [R + 'recortados_dataset_captioned/recortados_000.zip',
                                          R + 'recortados_dataset_captioned/recortados_001.zip'], COUNTS[name])
        elif name == 'parents':
            remote_zip_source(name, dst, [R + 'parents_dataset_captioned/parents_dataset.zip'], COUNTS[name])
        elif name == 'inscene':
            inscene_source(name, dst, COUNTS[name])
        else:
            sub = {'poxima_v2': 'ds2_poxima_v2', 'comikontext': 'ds3_comikontext',
                   'mega4': 'ds4_contexto_curado'}[name]
            ab_dirs_source(name, dst, pcg / sub, COUNTS[name])


if __name__ == '__main__':
    main()
