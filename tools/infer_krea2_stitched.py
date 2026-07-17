"""Repaint inference for the TRUE (stitched) IC-LoRA arm on plain Krea 2.

The LoRA is a plain t2i adapter trained on [ref|target] canvases. At inference
we generate the full canvas while locking the LEFT half to the reference at
every Euler step (flow-matching repaint: x_t = (1-t)*ref + t*noise0), so the
right half becomes "the next scene" conditioned purely through in-context
attention — no reference machinery involved.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils.common  # noqa: E402,F401  (import before ComfyUI shadows utils)

import toml  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tools.infer_reference_adapter import (  # noqa: E402
    decode_and_save,
    normalize_runtime_config,
    offload_diffusion,
    scale_adapter,
    setup_diffusion_pipeline,
)
from tools.krea2_sampling import build_krea2_timesteps  # noqa: E402

CAPTION_PREFIX = (
    'Two consecutive story panels side by side. The right panel is the next scene '
    'continuing the left panel, keeping the same characters, colors, lighting and style. '
    'Right panel: '
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--panel-width', type=int, default=672)
    parser.add_argument('--panel-height', type=int, default=384)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--text-guidance', type=float, default=1.0)
    parser.add_argument('--adapter-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=76)
    parser.add_argument('--blocks-to-swap', type=int, default=None)
    parser.add_argument('--no-caption-prefix', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config) as handle:
        config = toml.load(handle)
    config['model'].setdefault('dtype', 'bfloat16')
    normalize_runtime_config(config)

    from models.krea2 import Krea2Pipeline
    pipeline = Krea2Pipeline(config)

    prompt = args.prompt if args.no_caption_prefix else CAPTION_PREFIX + args.prompt
    cfg = 2 if args.text_guidance != 1.0 else 1
    pipeline.prepare_sample_test(prompt, negative_prompt='', cfg=cfg)

    # Reference -> left-half latent (same preprocessing as training pairs).
    image = Image.open(args.reference).convert('RGB')
    image = image.resize((args.panel_width, args.panel_height), Image.LANCZOS)
    import torchvision.transforms.functional as TF
    pixels = TF.pil_to_tensor(image).to(torch.float32) / 127.5 - 1.0
    pixels = pixels.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
    vae = pipeline.get_vae()
    vae.load_model_if_needed()
    ref_latent = pipeline.vae_encode(pixels.to('cuda', pipeline.dtype)).float()
    from comfy import model_management
    model_management.unload_all_models()
    torch.cuda.empty_cache()

    sequential, _ = setup_diffusion_pipeline(pipeline, Path(args.adapter), config, args.blocks_to_swap)
    scaled = scale_adapter(pipeline, args.adapter_scale)
    print(f'Adapter scale {args.adapter_scale} on {scaled} PEFT modules')

    sc = pipeline.spatial_compression
    lat_h = args.panel_height // sc
    lat_w = args.panel_width // sc
    canvas_shape = (1, pipeline.channels, 1, lat_h, 2 * lat_w)

    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    noise0 = torch.randn(canvas_shape, generator=generator, device='cuda')
    latent = noise0.clone()
    ref_latent = ref_latent.to('cuda')

    patch = int(pipeline.diffusion_model.patch)
    tokens = (lat_h // patch) * (2 * lat_w // patch)
    schedule, mu = build_krea2_timesteps(tokens, args.steps)
    print(f'Canvas {2 * args.panel_width}x{args.panel_height}, tokens={tokens}, mu={mu:.6f}')

    conds = tuple(v.to('cuda') for v in pipeline.conds)
    unconds = tuple(v.to('cuda') for v in getattr(pipeline, 'unconds', ()))

    def lock_left(x, t):
        x = x.clone()
        x[..., :lat_w] = (1.0 - t) * ref_latent + t * noise0[..., :lat_w]
        return x

    latent = lock_left(latent, schedule[0])
    for current, next_value in tqdm(list(zip(schedule[:-1], schedule[1:])), desc='Stitched repaint'):
        timestep = latent.new_full((1,), current)
        if args.text_guidance == 1.0:
            velocity = sequential((latent, timestep, *conds)).float()
        else:
            neg = sequential((latent, timestep, *unconds)).float()
            pos = sequential((latent, timestep, *conds)).float()
            velocity = neg + args.text_guidance * (pos - neg)
        latent = latent + (next_value - current) * velocity
        latent = lock_left(latent, next_value)

    offload_diffusion(pipeline, sequential)
    decode_and_save(pipeline, latent, args.output)
    right = Image.open(args.output).crop((args.panel_width, 0, 2 * args.panel_width, args.panel_height))
    right.save(args.output.with_name(args.output.stem + '_right.png'))
    print(f'Saved {args.output} (+ _right crop)')


if __name__ == '__main__':
    main()
