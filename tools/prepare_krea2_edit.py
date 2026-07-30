#!/usr/bin/env python3
"""Normaliza os datasets da saga KREA2-edit para o layout do loader multi_ref.

Layout de saída (por fonte):
    <dst>/<fonte>/target/<stem>.jpg   alvo (B)
    <dst>/<fonte>/target/<stem>.txt   caption do alvo
    <dst>/<fonte>/refs/<stem>_1.jpg   referência (A) — sufixo _1 = contrato N=1

Regras:
- long side > MAX_SIDE -> downscale (LANCZOS); tudo vira RGB JPEG q95.
- pares incompletos / imagens corrompidas / captions vazias -> rejeitados e logados.
- nada de avaliação de conteúdo aqui: só formato.

Uso:
    prepare_krea2_edit.py recortados <src_extracted> <dst>
    prepare_krea2_edit.py pairs_ab   <src_dir> <dst>          # genérico *_A/*_B
    prepare_krea2_edit.py pico       <pico_dir> <dst>         # selected.jsonl + edited/ + source/
    prepare_krea2_edit.py inscene    <parquet_dir> <dst>
"""
import argparse
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
MAX_SIDE = 1024
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_rgb(src) -> Image.Image | None:
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception:
        return None


def save_norm(img: Image.Image, dst: Path):
    w, h = img.size
    side = max(w, h)
    if side > MAX_SIDE:
        scale = MAX_SIDE / side
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    img.save(dst, 'JPEG', quality=95)


def process_one(job):
    """job = (target_src, ref_src, caption, stem, target_dir, refs_dir)
    target_src/ref_src podem ser paths ou bytes."""
    target_src, ref_src, caption, stem, target_dir, refs_dir = job
    t = load_rgb(io.BytesIO(target_src) if isinstance(target_src, bytes) else target_src)
    r = load_rgb(io.BytesIO(ref_src) if isinstance(ref_src, bytes) else ref_src)
    if t is None or r is None:
        return (stem, 'imagem corrompida/ilegível')
    caption = (caption or '').strip()
    if not caption:
        return (stem, 'caption vazia')
    try:
        save_norm(t, Path(target_dir) / f'{stem}.jpg')
        save_norm(r, Path(refs_dir) / f'{stem}_1.jpg')
        (Path(target_dir) / f'{stem}.txt').write_text(caption, encoding='utf-8')
    except Exception as e:
        return (stem, f'erro ao salvar: {e}')
    return None


def run_jobs(jobs, dst: Path, name: str, workers: int = 8):
    target_dir = dst / 'target'
    refs_dir = dst / 'refs'
    target_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(t, r, c, s, str(target_dir), str(refs_dir)) for t, r, c, s in jobs]
    rejects = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(process_one, jobs, chunksize=16):
            done += 1
            if res is not None:
                rejects.append(res)
            if done % 1000 == 0:
                print(f'  {name}: {done}/{len(jobs)}', flush=True)
    ok = len(jobs) - len(rejects)
    print(f'{name}: {ok} pares ok, {len(rejects)} rejeitados')
    if rejects:
        with open(dst / 'rejects.log', 'w') as f:
            for stem, why in rejects:
                f.write(f'{stem}\t{why}\n')
    return ok


def find_pairs_ab(src: Path):
    """Genérico: <stem>_A.<ext> (+ opcional _A.txt) / <stem>_B.<ext> + _B.txt."""
    files = [p for p in src.rglob('*') if p.is_file()]
    imgs = {}
    txts = {}
    for p in files:
        if p.suffix.lower() in IMG_EXTS:
            imgs[p.stem] = p
        elif p.suffix.lower() == '.txt':
            txts[p.stem] = p
    jobs = []
    missing = 0
    for stem_b, img_b in sorted(imgs.items()):
        if not stem_b.endswith('_B'):
            continue
        base = stem_b[:-2]
        img_a = imgs.get(base + '_A')
        txt_b = txts.get(stem_b)
        if img_a is None or txt_b is None:
            missing += 1
            continue
        caption = txt_b.read_text(encoding='utf-8', errors='replace')
        jobs.append((img_b, img_a, caption, base.replace('/', '_'), None))
    jobs = [(t, r, c, s) for t, r, c, s, _ in jobs]
    print(f'pares A/B encontrados: {len(jobs)} (incompletos: {missing})')
    return jobs


def cmd_recortados(args):
    jobs = find_pairs_ab(Path(args.src))
    run_jobs(jobs, Path(args.dst), 'recortados', args.workers)


def cmd_pairs_ab(args):
    jobs = find_pairs_ab(Path(args.src))
    run_jobs(jobs, Path(args.dst), Path(args.dst).name, args.workers)


def cmd_pico(args):
    src = Path(args.src)
    jobs = []
    missing_edit = missing_src = 0
    with open(src / 'selected.jsonl') as f:
        for line in f:
            d = json.loads(line)
            uid = d['uid']
            edited = src / 'edited' / f'{uid}.png'
            ext = d['src_url'].rsplit('.', 1)[-1].lower()
            if ext not in ('jpg', 'jpeg', 'png'):
                ext = 'jpg'
            source = src / 'source' / f'{uid}.{ext}'
            if not edited.exists() or edited.stat().st_size == 0:
                missing_edit += 1
                continue
            if not source.exists() or source.stat().st_size == 0:
                missing_src += 1
                continue
            jobs.append((edited, source, d['text'], uid))
    print(f'pico: {len(jobs)} pares completos (sem editada: {missing_edit}, sem fonte: {missing_src})')
    run_jobs(jobs, Path(args.dst), 'pico', args.workers)


def cmd_ab_dirs(args):
    """Formato mega2/3/4: <a_dir>/<id>.<ext> + <b_dir>/<id>.<ext>.

    Caption: .txt ao lado da imagem B; fallback em captions.jsonl
    ({"video": "images_B/<id>.jpg", "caption": ...}, entradas com error são
    ignoradas)."""
    src = Path(args.src)
    a_dir = src / args.a_dir
    b_dir = src / args.b_dir
    caps_jsonl = {}
    jsonl = src / 'captions.jsonl'
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cap = (d.get('caption') or '').strip()
                if not cap or d.get('error'):
                    continue
                caps_jsonl[Path(d['video']).stem] = cap
    a_imgs = {p.stem: p for p in a_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    b_imgs = {p.stem: p for p in b_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    jobs = []
    missing = 0
    prefix = args.prefix or src.name
    for stem in sorted(b_imgs):
        txt = b_dir / f'{stem}.txt'
        caption = None
        if txt.exists():
            caption = txt.read_text(encoding='utf-8', errors='replace').strip()
        if not caption:
            caption = caps_jsonl.get(stem)
        if stem in a_imgs and caption:
            jobs.append((b_imgs[stem], a_imgs[stem], caption, f'{prefix}_{stem}'))
        else:
            missing += 1
    print(f'ab_dirs: {len(jobs)} pares completos (incompletos/sem caption: {missing})')
    run_jobs(jobs, Path(args.dst), Path(args.dst).name, args.workers)


def cmd_inscene(args):
    import pyarrow.parquet as pq
    src = Path(args.src)
    jobs = []
    for pf in sorted(src.rglob('*.parquet')):
        table = pq.read_table(pf)
        cols = table.column_names
        for row in table.to_pylist():
            ctrl = row['control_image']['bytes'] if isinstance(row['control_image'], dict) else row['control_image']
            tgt = row['target_image']['bytes'] if isinstance(row['target_image'], dict) else row['target_image']
            split = pf.stem.split('-')[0]  # train / validation
            stem = f"inscene_{split}_{row.get('image_id') or ''}_{len(jobs):04d}".replace('/', '_')
            jobs.append((tgt, ctrl, row['prompt'], stem))
    print(f'inscene: {len(jobs)} linhas ({cols})')
    run_jobs(jobs, Path(args.dst), 'inscene', args.workers)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in [('recortados', cmd_recortados), ('pairs_ab', cmd_pairs_ab),
                     ('pico', cmd_pico), ('inscene', cmd_inscene),
                     ('ab_dirs', cmd_ab_dirs)]:
        p = sub.add_parser(name)
        p.add_argument('src')
        p.add_argument('dst')
        p.add_argument('--workers', type=int, default=8)
        p.add_argument('--prefix', default=None)
        p.add_argument('--a-dir', default='images_A')
        p.add_argument('--b-dir', default='images_B')
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
