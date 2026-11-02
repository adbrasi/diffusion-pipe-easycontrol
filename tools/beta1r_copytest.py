#!/usr/bin/env python3
"""Diagnóstico de cópia do BETA1-R: N adapters × M ref_bias × K seeds num só processo.

Mesmo protocolo do tools/beta1r_sample.py (turbo fundida na GPU, 8 steps, CFG 1,
mu 1.15, reference_fit crop, grounding Qwen3-VL), mas o pipeline carrega UMA vez:
TE+VAE codificam textos/refs, o DiT+turbo sobe, e para cada adapter só os pesos
da LoRA são trocados (load_adapter_weights cobre todas as chaves).

--runs  "nome:bias" (ex.: step1000:1.0 step1000:0.99). nome vem de --adapter nome=caminho.
ref_bias = viés aditivo log(b) nos logits alvo→referência (contrato do
CtxRushKrea2MultiRefControlledApply); 1.0 = máscara booleana original.

Também guarda a predição x0 no 1º step (x0 = x_t − σ·v) para medir se o colapso
já está decidido em ruído alto (dHash x0₁ × ref).

Saída: <out>/<run>/<seed>/<case>.jpg, <out>/<run>/metrics.json, <out>/<run>/sheet.jpg
(só casos SFW held-out = copy_eval), <out>/summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import infer_reference_adapter as ira  # noqa: E402

COPY_THR = 10


def dhash_img(img, size=8):
    a = np.asarray(img.convert('L').resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def ham(a, b):
    return int((a != b).sum())


def to_pil(image):
    # igual ao torchvision.utils.save_image (o PNG do beta1r_sample.py)
    if image.ndim == 5:
        image = image[:, 0]
    if image.shape[-1] in (3, 4):
        image = image.movedim(-1, 1)
    x = image[0, :3].float().clamp(0, 1).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    return Image.fromarray(x.permute(1, 2, 0).cpu().numpy())


@torch.no_grad()
def denoise(pipeline, model, reference, conds, shape, steps, seed, width, height, mu):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    latent = torch.randn(shape, generator=gen, device='cuda').to('cuda')
    reference = reference.to('cuda')
    patch = int(pipeline.diffusion_model.patch)
    seq = (shape[-2] // patch) * (shape[-1] // patch)
    schedule, _ = ira.build_krea2_timesteps(seq, steps, min_resolution=256, max_resolution=1280,
                                            spatial_compression=pipeline.spatial_compression,
                                            patch_size=patch, y1=0.5, y2=1.15, mu=mu)
    x0_first = None
    for cur, nxt in zip(schedule[:-1], schedule[1:]):
        t = latent.new_full((latent.shape[0],), cur)
        v = ira.call_model(model, latent, t, conds, reference)
        if x0_first is None:
            x0_first = (latent - cur * v).cpu()
        latent = latent + (nxt - cur) * v
    return latent.cpu(), x0_first


def sheet(run_dir, cases, seeds, out, h=160):
    rows = []
    for c in cases:
        ims = [Image.open(c['reference']).convert('RGB'), Image.open(c['target']).convert('RGB')]
        ims += [Image.open(run_dir / str(s) / f"{c['name']}.jpg").convert('RGB') for s in seeds]
        ims = [i.resize((max(1, round(i.width * h / i.height)), h)) for i in ims]
        canvas = Image.new('RGB', (sum(i.width for i in ims), h))
        x = 0
        for i in ims:
            canvas.paste(i, (x, 0))
            x += i.width
        rows.append(canvas)
    head = 20
    W = max(r.width for r in rows)
    grid = Image.new('RGB', (W, head + h * len(rows)), 'white')
    ImageDraw.Draw(grid).text((4, 4), f"{run_dir.name} | cols: ref, target, seeds {seeds}", fill='black')
    for k, r in enumerate(rows):
        grid.paste(r, (0, head + k * h))
    grid.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--adapter', action='append', required=True, help='nome=caminho')
    ap.add_argument('--runs', nargs='+', required=True, help='nome:bias')
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--turbo-lora', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[76])
    ap.add_argument('--png-run', default=None, help='run/seed que também salva PNG (sanidade)')
    args = ap.parse_args()

    adapters = dict(a.split('=', 1) for a in args.adapter)
    runs = [(r.split(':')[0], float(r.split(':')[1])) for r in args.runs]
    cases = json.loads(args.cases.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    ref_imgs = {c['name']: Image.open(c['reference']).convert('RGB') for c in cases}
    tgt_imgs = {c['name']: Image.open(c['target']).convert('RGB') for c in cases if c.get('target')}
    ref_h = {k: dhash_img(v) for k, v in ref_imgs.items()}
    tgt_h = {k: dhash_img(v) for k, v in tgt_imgs.items()}

    config = ira.load_raw_config(args.config)
    sargs = argparse.Namespace(turbo_lora=args.turbo_lora, krea_variant='auto', steps=None,
                               text_guidance=None, mu=None)
    ira.resolve_sampling_args(config, sargs)
    first_file = ira.find_adapter_file(Path(adapters[runs[0][0]]))
    ira.validate_contract(config, ira.read_metadata(first_file), allow_mismatch=False)
    ira.normalize_runtime_config(config)
    pipeline = ira.create_pipeline(config)

    enc = []
    for c in cases:
        pipeline.prepare_sample_test(c['prompt'], negative_prompt='', cfg=1,
                                     control_files=[str(c['reference'])])
        conds = tuple(v.cpu() for v in pipeline.conds)
        ref = ira.encode_reference(pipeline, Path(c['reference']), c['width'], c['height'], 'crop')
        enc.append((conds, ref))
    from comfy import model_management
    model_management.unload_all_models()

    pipeline.load_diffusion_model()
    pipeline.diffusion_model.to('cuda')
    print(f'turbo fundida em {ira.apply_turbo_lora(pipeline, args.turbo_lora)} modulos', flush=True)
    pipeline.configure_adapter(config['adapter'])
    pipeline.diffusion_model.eval()
    sequential = torch.nn.Sequential(*pipeline.to_layers())
    sequential.eval()
    init_layer = sequential[0]
    assert hasattr(init_layer, 'ref_bias'), 'initial layer sem ref_bias'

    prepared = []
    for c, (conds, ref) in zip(cases, enc):
        hw = (c['height'] // pipeline.spatial_compression, c['width'] // pipeline.spatial_compression)
        shape = (1, pipeline.channels, 1, *hw) if pipeline.is_video_vae else (1, pipeline.channels, *hw)
        reference = pipeline.prepare_reference_latents(ref, torch.zeros(shape, dtype=ref.dtype),
                                                       timestep_quantile=0.5)
        prepared.append((tuple(v.to('cuda') for v in conds), reference, shape))

    results = {}  # run -> list of (seed, case, final, x0)
    loaded = None
    for name, bias in runs:
        if loaded != name:
            f = ira.find_adapter_file(Path(adapters[name]))
            ira.validate_contract(config, ira.read_metadata(f), allow_mismatch=False)
            pipeline.load_adapter_weights(f)
            pipeline.diffusion_model.to('cuda')
            loaded = name
        init_layer.ref_bias = bias
        key = f'{name}_b{bias:g}'
        t0 = time.time()
        res = []
        for seed in args.seeds:
            for c, (conds, reference, shape) in zip(cases, prepared):
                lat, x0 = denoise(pipeline, sequential, reference, conds, shape, sargs.steps, seed,
                                  c['width'], c['height'], sargs.mu)
                res.append((seed, c['name'], lat, x0))
        results[key] = res
        print(f'RUN {key}: {len(res)} amostras em {time.time() - t0:.0f}s', flush=True)

    ira.offload_diffusion(pipeline, sequential)
    del sequential, init_layer
    torch.cuda.empty_cache()
    vae = pipeline.get_vae()
    vae.load_model_if_needed()
    by_name = {c['name']: c for c in cases}
    summary = {'protocol': {'steps': sargs.steps, 'cfg': 1.0, 'mu': sargs.mu, 'seeds': args.seeds,
                            'copy_threshold_dhash': COPY_THR, 'adapters': adapters,
                            'note': 'copy = dHash64(gen, ref) <= 10; x0_1 = predicao x0 no 1o step'},
                'runs': {}}
    for key, res in results.items():
        run_dir = args.out / key
        per = []
        for seed, cname, lat, x0 in res:
            img = to_pil(pipeline.vae_decode(lat.to('cuda')))
            x0img = to_pil(pipeline.vae_decode(x0.to('cuda')))
            d = run_dir / str(seed)
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f'{cname}.jpg', quality=88)
            if args.png_run and f'{key}/{seed}' == args.png_run:
                img.save(d / f'{cname}.png')
            gh, xh = dhash_img(img), dhash_img(x0img)
            c = by_name[cname]
            per.append({'seed': seed, 'case': cname, 'copy_eval': bool(c.get('copy_eval', False)),
                        'dhash_gen_ref': ham(gh, ref_h[cname]),
                        'dhash_gen_target': ham(gh, tgt_h[cname]) if cname in tgt_h else None,
                        'dhash_x0_1_ref': ham(xh, ref_h[cname]),
                        'dhash_x0_1_gen': ham(xh, gh),
                        'copy': ham(gh, ref_h[cname]) <= COPY_THR})

        def agg(rows):
            if not rows:
                return None
            return {'n': len(rows), 'copies': sum(r['copy'] for r in rows),
                    'copy_rate': sum(r['copy'] for r in rows) / len(rows),
                    'mean_dhash_gen_ref': float(np.mean([r['dhash_gen_ref'] for r in rows])),
                    'mean_dhash_gen_target': float(np.mean([r['dhash_gen_target'] for r in rows])),
                    'mean_dhash_gen_ref_noncopy': (float(np.mean([r['dhash_gen_ref'] for r in rows if not r['copy']]))
                                                   if any(not r['copy'] for r in rows) else None),
                    'copied': sorted(f"{r['case']}@{r['seed']}" for r in rows if r['copy'])}
        m = {'heldout': agg([p for p in per if p['copy_eval']]),
             'beta1_examples': agg([p for p in per if not p['copy_eval']]),
             'samples': per}
        (run_dir / 'metrics.json').write_text(json.dumps(m, indent=1))
        sheet(run_dir, [c for c in cases if c.get('copy_eval')], args.seeds, run_dir / 'sheet.jpg')
        summary['runs'][key] = {'heldout': m['heldout'], 'beta1_examples': m['beta1_examples']}
        print(f"SUMMARY {key} heldout {m['heldout']['copies']}/{m['heldout']['n']} "
              f"beta1 {m['beta1_examples']['copies']}/{m['beta1_examples']['n']}", flush=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('COPYTEST OK', args.out)


if __name__ == '__main__':
    main()
