#!/usr/bin/env python3
"""Samples + métrica de cópia do BETA1-R num checkpoint (carrega o pipeline UMA vez).

Protocolo de sample do beta1 (krea2_saga_supervisor.run_samples): LoRA turbo
oficial fundida no base, 8 steps, CFG 1.0, mu 1.15, seed 76, reference_fit crop,
grounding Qwen3-VL na ref. Mesmo caminho de tools/infer_reference_adapter.py.

cases.json: [{"name", "reference", "prompt", "width", "height", "target"?: gt,
              "copy_eval"?: bool}]
Saída: <out>/<name>.png, <out>/grid.jpg, <out>/metrics.json

Métrica de cópia (triagem, não veredito): dHash 64-bit gerado×ref e gerado×alvo.
copy = dist(gerado, ref) <= COPY_THR (quase a mesma imagem que a referência).
copy_rate = fração dos casos copy_eval com copy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import infer_reference_adapter as ira  # noqa: E402

COPY_THR = 10


def dhash(path, size=8):
    img = Image.open(path).convert('L').resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hamming(a, b):
    return int((dhash(a) != dhash(b)).sum())


def make_grid(rows, out, h=256):
    tiles = []
    for row in rows:
        ims = []
        for p in row:
            im = Image.open(p).convert('RGB') if p and Path(p).exists() else Image.new('RGB', (h, h))
            ims.append(im.resize((max(1, round(im.width * h / im.height)), h)))
        w = sum(i.width for i in ims)
        canvas = Image.new('RGB', (w, h))
        x = 0
        for i in ims:
            canvas.paste(i, (x, 0))
            x += i.width
        tiles.append(canvas)
    W = max(t.width for t in tiles)
    grid = Image.new('RGB', (W, h * len(tiles)), 'white')
    for k, t in enumerate(tiles):
        grid.paste(t, (0, k * h))
    grid.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, required=True)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--turbo-lora', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=76)
    ap.add_argument('--blocks-to-swap', type=int, default=0)
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    config = ira.load_raw_config(args.config)
    sargs = argparse.Namespace(turbo_lora=args.turbo_lora, krea_variant='auto', steps=None,
                               text_guidance=None, mu=None)
    ira.resolve_sampling_args(config, sargs)
    adapter_file = ira.find_adapter_file(args.adapter)
    ira.validate_contract(config, ira.read_metadata(adapter_file), allow_mismatch=False)
    ira.normalize_runtime_config(config)
    pipeline = ira.create_pipeline(config)

    # Fase A: textos (TE) + refs (VAE)
    enc = []
    for c in cases:
        pipeline.prepare_sample_test(c['prompt'], negative_prompt='', cfg=1,
                                     control_files=[str(c['reference'])])
        conds = tuple(v.cpu() for v in pipeline.conds)
        ref = ira.encode_reference(pipeline, Path(c['reference']), c['width'], c['height'], 'crop')
        enc.append((conds, ref))
    from comfy import model_management
    model_management.unload_all_models()

    # Fase B: DiT + turbo + adapter
    sequential, _ = ira.setup_diffusion_pipeline(pipeline, adapter_file, config, args.blocks_to_swap,
                                                 turbo_lora=args.turbo_lora)
    latents = []
    for c, (conds, ref) in zip(cases, enc):
        hw = (c['height'] // pipeline.spatial_compression, c['width'] // pipeline.spatial_compression)
        shape = (1, pipeline.channels, 1, *hw) if pipeline.is_video_vae else (1, pipeline.channels, *hw)
        reference = pipeline.prepare_reference_latents(ref, torch.zeros(shape, dtype=ref.dtype),
                                                       timestep_quantile=0.5)
        pipeline.conds = conds
        lat = ira.denoise(pipeline, sequential, reference, shape, sargs.steps, args.seed,
                          sargs.text_guidance, 1.0, None, c['width'], c['height'], sargs.mu,
                          256, 1280, 0.5, 1.15, 'cuda')
        latents.append(lat.cpu())
    ira.offload_diffusion(pipeline, sequential)
    rows, per = [], []
    for c, lat in zip(cases, latents):
        out = args.out / f"{c['name']}.png"
        ira.decode_and_save(pipeline, lat.to('cuda'), out)
        d_ref = hamming(out, c['reference'])
        d_tgt = hamming(out, c['target']) if c.get('target') else None
        per.append({'name': c['name'], 'dhash_gen_ref': d_ref, 'dhash_gen_target': d_tgt,
                    'dhash_ref_target': hamming(c['reference'], c['target']) if c.get('target') else None,
                    'copy': d_ref <= COPY_THR, 'copy_eval': bool(c.get('copy_eval', False))})
        rows.append([c['reference'], str(out), c.get('target')])
    ev = [p for p in per if p['copy_eval']]
    metrics = {'adapter': str(adapter_file), 'seed': args.seed, 'steps': sargs.steps,
               'copy_threshold_dhash': COPY_THR,
               'copy_rate': (sum(p['copy'] for p in ev) / len(ev)) if ev else None,
               'mean_dhash_gen_ref': (sum(p['dhash_gen_ref'] for p in ev) / len(ev)) if ev else None,
               'n_copy_eval': len(ev), 'cases': per}
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    make_grid(rows, args.out / 'grid.jpg')
    print(f"COPY_RATE {metrics['copy_rate']} mean_dhash_gen_ref {metrics['mean_dhash_gen_ref']}")
    print('SAMPLES OK', args.out)


if __name__ == '__main__':
    main()
