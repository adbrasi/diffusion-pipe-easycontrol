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
import shutil
import sys
import time
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


def _norm_job(args):
    data, dst = args
    img = load_rgb(io.BytesIO(data))
    if img is None:
        return dst, 'imagem ilegível'
    save_norm(img, Path(dst))
    return dst, None


def remote_zip_source(name, dst, repo_files, n):
    """UM GET em streaming por zip (range reads estouram o rate limit do HF).

    Passo 1: normaliza TODO par completo do zip em <dst>/_staging (mesma regra
    de find_pairs_ab: <stem>_B.<img> + <stem>_B.txt + <stem>_A.<img>; stems
    repetidos entre zips: o último vence). Passo 2: mesma amostragem seed 42
    sobre os pares ordenados; os escolhidos vão para target/refs, o resto sai.
    """
    import requests
    from concurrent.futures import ProcessPoolExecutor
    from stream_unzip import stream_unzip
    token = os.environ['HF_TOKEN']
    stage = dst / '_staging'
    (stage / 'img').mkdir(parents=True, exist_ok=True)
    (stage / 'txt').mkdir(parents=True, exist_ok=True)
    pending = []
    with ProcessPoolExecutor(48) as ex:
        for path in repo_files:
            repo = '/'.join(path.split('/')[1:3]); fname = path.split('/', 3)[3]
            url = f'https://huggingface.co/datasets/{repo}/resolve/main/{fname}'
            for attempt in range(20):
                r = requests.get(url, headers={'Authorization': f'Bearer {token}'}, stream=True, timeout=120)
                if r.status_code == 429:
                    print(f'{name}: 429, aguardando 60s', flush=True); time.sleep(60); continue
                r.raise_for_status(); break
            count = 0
            for member, _size, chunks in stream_unzip(r.iter_content(1 << 20)):
                member = member.decode('utf-8', errors='replace')
                data = b''.join(chunks)
                p = Path(member)
                if p.suffix.lower() in IMG_EXTS and p.stem[-2:] in ('_A', '_B'):
                    out = stage / 'img' / f'{p.stem}.jpg'
                    pending.append(ex.submit(_norm_job, (data, str(out))))
                elif p.suffix.lower() == '.txt' and p.stem.endswith('_B'):
                    (stage / 'txt' / f'{p.stem}.txt').write_bytes(data)
                count += 1
                if count % 2000 == 0:
                    print(f'  {name}: {fname} {count} membros', flush=True)
                if len(pending) > 2000:
                    pending = [f for f in pending if not f.done()]
            print(f'{name}: {fname} lido ({count} membros)', flush=True)
        bad = set()
        for f in pending:
            d, err = f.result()
            if err:
                bad.add(Path(d).stem)
    imgs = {p.stem for p in (stage / 'img').iterdir()} - bad
    txts = {p.stem for p in (stage / 'txt').iterdir()}
    pairs = [(s[:-2],) for s in imgs if s.endswith('_B') and s[:-2] + '_A' in imgs and s in txts]
    print(f'{name}: {len(pairs)} pares completos', flush=True)
    chosen = sample(pairs, n, name)
    (dst / 'target').mkdir(parents=True, exist_ok=True)
    (dst / 'refs').mkdir(parents=True, exist_ok=True)
    rejects = 0
    for (base,) in chosen:
        cap = (stage / 'txt' / f'{base}_B.txt').read_text(encoding='utf-8', errors='replace').strip()
        if not cap:
            rejects += 1
            continue
        stem = f'{name}_{base}'
        os.replace(stage / 'img' / f'{base}_B.jpg', dst / 'target' / f'{stem}.jpg')
        os.replace(stage / 'img' / f'{base}_A.jpg', dst / 'refs' / f'{stem}_1.jpg')
        (dst / 'target' / f'{stem}.txt').write_text(cap, encoding='utf-8')
    shutil.rmtree(stage)
    print(f'{name}: {len(chosen) - rejects} ok, {rejects} rejeitados (caption vazia)', flush=True)


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
