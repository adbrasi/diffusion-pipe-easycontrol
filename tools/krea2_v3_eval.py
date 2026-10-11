#!/usr/bin/env python3
"""Krea 2 next-scene eval through a headless STOCK ComfyUI (core nodes only, --disable-all-custom-nodes).

Graph = what the user runs: UNETLoader(raw fp8) [+ Turbo LoRA] [+ our LoRA @ strength via LoraLoaderModelOnly]
-> TextEncodeQwenImageEditPlus(clip, prompt, vae, image1) -> FluxKontextMultiReferenceLatentMethod(index_timestep_zero)
-> CFGGuider -> SamplerCustomAdvanced(euler, Krea2 sigmas). Reference = node's own ~1MP rule (node_1mp).

Cases file (JSON list): {"id", "kind", "ref", "target"|null, "prompt"}; kind in
train_caption | train_altered | heldout | short. Output size = training AR bucket of the target
(or of the ref when there is no target) at ~1MP, sides /16.

Columns (--col, repeatable): "label=native" | "label=<name>,adapter=<.../adapter_model.safetensors>,strength=0.7"
optional keys: shuffle=1 (reference of the next case), variant=Turbo|Raw|TurboOfficial.
Default variant from --variant. Seeds: --seeds 76 (several -> more rows).
Writes grid.png (A | B | columns), images, results.json (+ copy metric: corr(gen, A) vs corr(gen, B) on 64px gray).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.krea2_sampling import build_krea2_timesteps  # noqa: E402

AR_BUCKETS = [(1, 2), (9, 16), (2, 3), (3, 4), (1, 1), (4, 3), (3, 2), (16, 9), (2, 1), (7, 3)]


def bucket_dims(path, area=1024 * 1024):
    """Training AR bucket (same list as the dataset config) at ~area pixels, sides /16."""
    with Image.open(path) as im:
        ar = im.width / im.height
    b = min((w / h for w, h in AR_BUCKETS), key=lambda r: abs(math.log(ar / r)))
    return round(math.sqrt(area * b) / 16) * 16, round(math.sqrt(area / b) / 16) * 16


def parse_col(spec, default_variant):
    col = dict(kv.split('=', 1) for kv in spec.split(','))
    col.setdefault('variant', default_variant)
    col['strength'] = float(col.get('strength', 1.0))
    col['shuffle'] = col.get('shuffle', '0') in ('1', 'true', 'yes')
    if col.get('label') == 'native':
        col['adapter'] = None
    return col


def graph(prompt, width, height, seed, reference, variant, lora_name, strength, prefix, models,
          raw_steps=28, raw_cfg=4.0):
    turbo = variant in ('Turbo', 'TurboOfficial')
    steps = 8 if turbo else raw_steps
    cfg = 1.0 if turbo else raw_cfg
    mu = 1.15 if turbo else None
    sigmas, _ = build_krea2_timesteps((width // 16) * (height // 16), steps, mu=mu)
    nodes = {}

    def add(name, node_type, **inputs):
        nodes[name] = dict(class_type=node_type, inputs=inputs)
        return [name, 0]

    base = models['turbo_full'] if variant == 'TurboOfficial' else models['raw']
    model = add('1', 'UNETLoader', unet_name=base, weight_dtype='default')
    clip = add('2', 'CLIPLoader', clip_name=models['clip'], type='krea2', device='default')
    vae = add('3', 'VAELoader', vae_name=models['vae'])
    image = add('4', 'LoadImage', image=reference)
    if variant == 'Turbo':
        model = add('5', 'LoraLoaderModelOnly', model=model, lora_name=models['turbo_lora'], strength_model=1.0)
    if lora_name:
        model = add('6', 'LoraLoaderModelOnly', model=model, lora_name=lora_name, strength_model=strength)
    pos = add('7', 'TextEncodeQwenImageEditPlus', clip=clip, prompt=prompt, vae=vae, image1=image)
    neg = add('8', 'TextEncodeQwenImageEditPlus', clip=clip, prompt='', vae=vae, image1=image)
    pos = add('9', 'FluxKontextMultiReferenceLatentMethod', conditioning=pos, reference_latents_method='index_timestep_zero')
    neg = add('10', 'FluxKontextMultiReferenceLatentMethod', conditioning=neg, reference_latents_method='index_timestep_zero')
    guider = add('11', 'CFGGuider', model=model, positive=pos, negative=neg, cfg=cfg)
    latent = add('12', 'EmptyLatentImage', width=width, height=height, batch_size=1)
    noise = add('13', 'RandomNoise', noise_seed=seed)
    sampler = add('14', 'KSamplerSelect', sampler_name='euler')
    sig = add('15', 'ManualSigmas', sigmas=', '.join(f'{v:.12f}' for v in sigmas))
    samples = add('16', 'SamplerCustomAdvanced', noise=noise, guider=guider, sampler=sampler, sigmas=sig, latent_image=latent)
    pixels = add('17', 'VAEDecode', samples=samples, vae=vae)
    add('18', 'SaveImage', images=pixels, filename_prefix=prefix)
    return nodes


def wait_ready(url, timeout=300):
    deadline = time.time() + timeout
    while True:
        try:
            requests.get(url + '/system_stats', timeout=5).raise_for_status()
            return
        except requests.RequestException:
            if time.time() > deadline:
                raise
            time.sleep(2)


def execute(prompt, destination, url, output_root):
    r = requests.post(url + '/prompt', json={'prompt': prompt}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f'ComfyUI rejected the prompt: {r.text[:3000]}')
    res = r.json()
    if res.get('node_errors'):
        raise RuntimeError(json.dumps(res))
    pid = res['prompt_id']
    start = time.time()
    while time.time() - start < 1800:
        hist = requests.get(url + '/history/' + pid, timeout=30).json()
        if pid in hist:
            entry = hist[pid]
            if entry.get('status', {}).get('status_str') == 'error':
                raise RuntimeError(json.dumps(entry['status'])[:4000])
            imgs = entry.get('outputs', {}).get('18', {}).get('images', [])
            if imgs:
                row = imgs[0]
                shutil.copyfile(output_root / row['subfolder'] / row['filename'], destination)
                return
        time.sleep(1)
    raise TimeoutError(pid)


def gray64(img_or_path, size):
    im = Image.open(img_or_path) if not isinstance(img_or_path, Image.Image) else img_or_path
    im = im.convert('L')
    # crop-fit to the generated AR before comparing
    w, h = im.size
    ar = size[0] / size[1]
    if w / h > ar:
        nw = int(h * ar); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / ar); im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    a = np.asarray(im.resize((64, max(8, round(64 / ar))), Image.BILINEAR), dtype=np.float32)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def corr(a, b):
    return float((a * b).sum())


def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_grid(rows, col_titles, cell=300, text_w=300):
    f, fs = font(14), font(12)
    head = 30
    g = Image.new('RGB', (text_w + len(col_titles) * cell, head + len(rows) * cell), 'white')
    d = ImageDraw.Draw(g)
    for j, t in enumerate(col_titles):
        d.text((text_w + j * cell + 6, 8), t, fill='black', font=f)
    for i, (label, cells) in enumerate(rows):
        y = head + i * cell
        d.line([(0, y), (g.width, y)], fill=(200, 200, 200))
        lines = textwrap.wrap(label, 44)[:20]
        d.multiline_text((6, y + 6), '\n'.join(lines), fill='black', font=fs)
        for j, im in enumerate(cells):
            if im is None:
                continue
            im = im.convert('RGB').copy()
            im.thumbnail((cell - 4, cell - 4))
            g.paste(im, (text_w + j * cell + 2, y + 2))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--col', action='append', required=True)
    ap.add_argument('--variant', default='Turbo', choices=('Turbo', 'Raw', 'TurboOfficial'))
    ap.add_argument('--seeds', default='76')
    ap.add_argument('--kinds', default=None, help='comma list of case kinds to keep')
    ap.add_argument('--ids', default=None, help='comma list of case ids to keep')
    ap.add_argument('--raw-steps', type=int, default=28)
    ap.add_argument('--raw-cfg', type=float, default=4.0)
    ap.add_argument('--comfy', type=Path, default=Path('/workspace/comfy/ComfyUI'))
    ap.add_argument('--python', default='/workspace/venv/bin/python')
    ap.add_argument('--models', type=Path, default=Path('/workspace/models/krea2'))
    ap.add_argument('--port', type=int, default=18819)
    ap.add_argument('--keep-server', action='store_true', help='reuse a server already listening on --port')
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    if args.kinds:
        cases = [c for c in cases if c['kind'] in args.kinds.split(',')]
    if args.ids:
        cases = [c for c in cases if c['id'] in args.ids.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    cols = [parse_col(s, args.variant) for s in args.col]
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='k2v3_'))
    for sub in ('input', 'output', 'loras'):
        (work / sub).mkdir()
    for c in cases:
        dst = work / 'input' / (c['id'] + '.png')
        Image.open(c['ref']).convert('RGB').save(dst)
        c['ref_name'] = dst.name
    for col in cols:
        if col.get('adapter'):
            src = Path(col['adapter']).resolve()
            name = hashlib.md5(str(src).encode()).hexdigest()[:10] + '.safetensors'
            if not (work / 'loras' / name).exists():
                (work / 'loras' / name).symlink_to(src)
            col['lora_name'] = 'loras/' + name
        else:
            col['lora_name'] = None
    (work / 'paths.yaml').write_text(
        f'krea2:\n  base_path: {args.models}\n  diffusion_models: diffusion_models\n'
        f'  text_encoders: text_encoders\n  vae: vae\n  loras: loras\n'
        f'k2v3:\n  base_path: {work}\n  loras: .\n')
    models = dict(raw='krea2_raw_fp8_scaled.safetensors', turbo_full='krea2_turbo_fp8_scaled.safetensors',
                  turbo_lora='krea2_turbo_lora_rank_64_bf16.safetensors', clip='qwen3vl_4b_bf16.safetensors',
                  vae='qwen_image_vae.safetensors')
    url = f'http://127.0.0.1:{args.port}'
    server = None
    log = (args.out / 'comfy_server.log').open('a')
    if not args.keep_server:
        server = subprocess.Popen(
            [args.python, 'main.py', '--listen', '127.0.0.1', '--port', str(args.port), '--disable-all-custom-nodes',
             '--extra-model-paths-config', str(work / 'paths.yaml'), '--input-directory', str(work / 'input'),
             '--output-directory', str(work / 'output')], cwd=args.comfy, stdout=log, stderr=subprocess.STDOUT)
    results, grid_rows = [], []
    t0 = time.time()
    try:
        wait_ready(url)
        for seed in seeds:
            for i, c in enumerate(cases):
                size_src = c.get('target') or c['ref']
                w, h = bucket_dims(size_src)
                ga = gray64(c['ref'], (w, h))
                gb = gray64(c['target'], (w, h)) if c.get('target') else None
                cells = [Image.open(c['ref']), Image.open(c['target']) if c.get('target') else None]
                for k, col in enumerate(cols):
                    ref = c['ref_name']
                    if col['shuffle']:  # next case with a DIFFERENT reference image
                        ref = next(cases[(i + d) % len(cases)]['ref_name'] for d in range(1, len(cases) + 1)
                                   if cases[(i + d) % len(cases)]['ref'] != c['ref'])
                    dest = args.out / f"{c['id']}_s{seed}_{col['label']}.png"
                    if not dest.exists():
                        g = graph(c['prompt'], w, h, seed, ref, col['variant'], col['lora_name'], col['strength'],
                                  f'k2v3_{k}', models, args.raw_steps, args.raw_cfg)
                        execute(g, dest, url, work / 'output')
                    gen = Image.open(dest)
                    gg = gray64(gen, (w, h))
                    r = dict(case=c['id'], kind=c['kind'], seed=seed, col=col['label'], variant=col['variant'],
                             adapter=col.get('adapter'), strength=col['strength'], shuffle=col['shuffle'],
                             width=w, height=h, corr_A=corr(gg, ga), corr_B=corr(gg, gb) if gb is not None else None,
                             file=dest.name)
                    results.append(r)
                    cells.append(gen)
                    print(f"[{time.time() - t0:6.0f}s] {c['id']} s{seed} {col['label']}: corrA={r['corr_A']:.3f}"
                          + (f" corrB={r['corr_B']:.3f}" if r['corr_B'] is not None else ''), flush=True)
                label = f"[{c['kind']}] {c['id']} seed {seed} {w}x{h}\n{c['prompt']}"
                grid_rows.append((label, cells))
    finally:
        if server:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
        log.close()
        shutil.rmtree(work, ignore_errors=True)
    titles = ['A / reference', 'B / ground truth'] + [
        f"{c['label']} ({c['variant']}{'' if c['lora_name'] is None else ' @' + str(c['strength'])}{' shuffled' if c['shuffle'] else ''})"
        for c in cols]
    # Split grids by kind so each stays readable.
    kinds = list(dict.fromkeys(label.split(']')[0][1:] for label, _ in grid_rows))
    for k in kinds:
        make_grid([r for r in grid_rows if r[0].startswith(f'[{k}]')], titles).save(args.out / f'grid_{k}.jpg', quality=88)
    summary = {}
    for col in cols:
        rs = [r for r in results if r['col'] == col['label']]
        summary[col['label']] = dict(
            corr_A=float(np.mean([r['corr_A'] for r in rs])),
            corr_B=float(np.mean([r['corr_B'] for r in rs if r['corr_B'] is not None] or [np.nan])))
    (args.out / 'results.json').write_text(json.dumps(dict(
        cases=str(args.cases), seeds=seeds, cols=[{k: v for k, v in c.items()} for c in cols],
        summary=summary, results=results), indent=1))
    print(json.dumps(summary, indent=1))
    print(f'done in {time.time() - t0:.0f}s -> {args.out}')


if __name__ == '__main__':
    main()
