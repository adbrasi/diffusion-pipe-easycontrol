#!/usr/bin/env python3
"""T2I puro no Krea 2 base — SEM referência, SEM adapter.

Baseline de qualidade: o mesmo stack de código do fork (fp8 + turbo LoRA +
schedule Euler oficial com mu), mas o pipeline base `krea2`, cuja sequência é
apenas [texto | target]. Nada de reference packing.

Uso:
  python tools/t2i_krea2_base.py --prompt "..." --output out.png \
      [--width 1152 --height 896 --seed 76 --steps 8 --guidance 1.0] \
      [--variant turbo|raw]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

MODELS = {
    'diffusion_model': '/workspace/models/krea2/diffusion_models/krea2_raw_fp8_scaled.safetensors',
    'vae': '/workspace/models/krea2/split_files/vae/qwen_image_vae.safetensors',
    'text_encoder': '/workspace/models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors',
    'turbo_lora': '/workspace/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--negative-prompt', default='')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--width', type=int, default=1152)
    ap.add_argument('--height', type=int, default=896)
    ap.add_argument('--seed', type=int, default=76)
    ap.add_argument('--variant', choices=('turbo', 'raw'), default='turbo')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--guidance', type=float, default=None)
    args = ap.parse_args()

    if args.variant == 'turbo':
        steps = args.steps or 8
        guidance = 1.0 if args.guidance is None else args.guidance
        mu = 1.15
    else:
        steps = args.steps or 28
        guidance = 5.5 if args.guidance is None else args.guidance
        mu = None  # derivado da resolução

    from models.krea2 import Krea2Pipeline
    from tools.infer_reference_adapter import apply_turbo_lora, decode_and_save
    from tools.krea2_sampling import build_krea2_timesteps

    config = {
        'model': {
            'type': 'krea2',
            'diffusion_model': MODELS['diffusion_model'],
            'vae': MODELS['vae'],
            'text_encoders': [{'path': MODELS['text_encoder'], 'type': 'krea2'}],
            'dtype': 'bfloat16',
            'diffusion_model_dtype': 'float8',
            'flux_shift': True,
        },
        'adapter': None,
    }
    pipeline = Krea2Pipeline(config)
    pipeline.prepare_sample_test(args.prompt, negative_prompt=args.negative_prompt,
                                 cfg=2 if guidance != 1.0 else 1)
    pipeline.load_diffusion_model()
    if args.variant == 'turbo':
        n = apply_turbo_lora(pipeline, Path(MODELS['turbo_lora']))
        print(f'turbo LoRA fundida em {n} modulos')
    pipeline.diffusion_model.eval()
    pipeline.diffusion_model.to('cuda')
    model = torch.nn.Sequential(*pipeline.to_layers()).eval()

    sc = pipeline.spatial_compression
    shape = (1, pipeline.channels, 1, args.height // sc, args.width // sc)
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    latent = torch.randn(shape, generator=generator, device='cuda')

    patch = int(pipeline.diffusion_model.patch)
    seq_len = (shape[-2] // patch) * (shape[-1] // patch)
    schedule, resolved_mu = build_krea2_timesteps(
        seq_len, steps, spatial_compression=sc, patch_size=patch, mu=mu)
    print(f'{args.variant}: {args.width}x{args.height}, steps={steps}, '
          f'guidance={guidance}, mu={resolved_mu:.4f}, tokens={seq_len}')

    conds = tuple(v.to('cuda') for v in pipeline.conds)
    unconds = tuple(v.to('cuda') for v in getattr(pipeline, 'unconds', ()))

    with torch.no_grad():
        for current, next_value in tqdm(list(zip(schedule[:-1], schedule[1:]))):
            t = latent.new_full((1,), current)
            v = model((latent, t, *conds)).float()
            if guidance != 1.0:
                v_u = model((latent, t, *unconds)).float()
                v = v_u + guidance * (v - v_u)
            latent = latent + (next_value - current) * v

    pipeline.diffusion_model.to('cpu')
    torch.cuda.empty_cache()
    decode_and_save(pipeline, latent, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
