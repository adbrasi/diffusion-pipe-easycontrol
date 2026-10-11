#!/usr/bin/env python3
"""Held-out eval of a krea2_native adapter through a headless STOCK ComfyUI (core nodes only).

Starts the user's ComfyUI with --disable-all-custom-nodes on its own port, runs the
k2ab_eval_stock graph (TextEncodeQwenImageEditPlus + index_timestep_zero, adapter via
LoraLoaderModelOnly) and stops the server so the GPU is free for training again.
Per pair: Turbo + adapter and Turbo + adapter with a shuffled reference (--raw adds Raw 28 steps).
Grid columns: A | B | Turbo | [Raw] | Turbo shuffled.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.k2ab_eval_stock import execute, graph  # noqa: E402

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp'}


def bucket_dims(path, area=1024 * 1024):
    with Image.open(path) as im:
        ar = im.width / im.height
    bucket = min((2 ** (i / 3) for i in range(-3, 4)), key=lambda r: abs(math.log(ar / r)))
    return round(math.sqrt(area * bucket) / 16) * 16, round(math.sqrt(area / bucket) / 16) * 16


def grid(rows, titles, cell=256):
    g = Image.new('RGB', (len(titles) * cell, len(rows) * cell + 28), 'white')
    draw = ImageDraw.Draw(g)
    for j, title in enumerate(titles):
        draw.text((j * cell + 8, 8), title, fill='black')
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            im = im.convert('RGB').copy()
            im.thumbnail((cell, cell))
            g.paste(im, (j * cell, i * cell + 28))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adapter', type=Path, required=True, help='.../stepN (contains adapter_model.safetensors)')
    ap.add_argument('--pairs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=3)
    ap.add_argument('--comfy', type=Path, default=Path('/workspace/comfy/ComfyUI'))
    ap.add_argument('--python', default='/workspace/comfy/.venv/bin/python')
    ap.add_argument('--models', type=Path, default=Path('/workspace/models/krea2'))
    ap.add_argument('--base', default='krea2_raw_fp8_scaled.safetensors')
    ap.add_argument('--port', type=int, default=18819)
    ap.add_argument('--seed', type=int, default=76)
    ap.add_argument('--raw', action='store_true', help='also generate Raw 28 steps (slow)')
    args = ap.parse_args()

    targets = sorted(p for p in (args.pairs / 'target').iterdir() if p.suffix.lower() in IMG_EXT)
    controls = {p.stem: p for p in (args.pairs / 'control').iterdir() if p.suffix.lower() in IMG_EXT}
    pairs = [(t.stem, controls[t.stem], t, t.with_suffix('.txt').read_text().strip())
             for t in targets if t.stem in controls][:args.limit]
    assert len(pairs) >= 2, 'need >= 2 pairs for the shuffled-reference column'

    name = '_'.join(args.adapter.parts[-2:])
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='k2eval_'))
    (work / 'input').mkdir()
    (work / 'output').mkdir()
    for _, control, _, _ in pairs:
        Image.open(control).convert('RGB').save(work / 'input' / (control.stem + '.png'))
    lora_root = args.adapter.parent
    (work / 'paths.yaml').write_text(
        f'krea2:\n  base_path: {args.models}\n  diffusion_models: diffusion_models\n'
        f'  text_encoders: text_encoders\n  vae: vae\n  loras: loras\n'
        f'run:\n  base_path: {lora_root}\n  loras: .\n')
    adapter_name = f'{args.adapter.name}/adapter_model.safetensors'
    url = f'http://127.0.0.1:{args.port}'
    log = (out / 'comfy_server.log').open('w')
    server = subprocess.Popen(
        [args.python, 'main.py', '--listen', '127.0.0.1', '--port', str(args.port), '--disable-all-custom-nodes',
         '--extra-model-paths-config', str(work / 'paths.yaml'), '--input-directory', str(work / 'input'),
         '--output-directory', str(work / 'output')], cwd=args.comfy, stdout=log, stderr=subprocess.STDOUT)
    rows, prompts = [], {}
    try:
        for i, (stem, control, target, prompt) in enumerate(pairs):
            w, h = bucket_dims(target)
            shuffled = pairs[(i + 1) % len(pairs)][1]
            row = dict(prompt=prompt, width=w, height=h, seed=args.seed)
            images = []
            variants = [('turbo', 'Turbo', control)] + ([('raw', 'Raw', control)] if args.raw else []) + \
                [('turbo_shuffled', 'Turbo', shuffled)]
            for tag, variant, ref in variants:
                dest = out / f'{stem}_{tag}.png'
                execute(graph(row, ref.stem + '.png', variant, adapter_name, f'{stem}_{tag}', base_model=args.base),
                        dest, url=url, output_root=work / 'output')
                images.append(Image.open(dest))
            rows.append([Image.open(control), Image.open(target)] + images)
            prompts[stem] = prompt
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    grid(rows, ['A / reference', 'B / ground truth', 'Turbo 8st + adapter'] + (['Raw 28st + adapter'] if args.raw else [])
         + ['Turbo, shuffled ref']).save(out / 'grid.png')
    (out / 'prompts.json').write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    (out / 'eval_config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f'grid: {out / "grid.png"}')


if __name__ == '__main__':
    main()
